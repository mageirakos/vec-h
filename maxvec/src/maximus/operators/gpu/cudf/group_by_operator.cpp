#include <thrust/device_vector.h>

#include <cudf/concatenate.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/reduction.hpp>
#include <maximus/gpu/cuda_api.hpp>
#include <maximus/operators/gpu/cudf/group_by_operator.hpp>
#include <typeinfo>

namespace maximus::cudf {

arrow::Status get_key_indices(const std::shared_ptr<arrow::Schema> &schema,
                              const std::vector<arrow::FieldRef> &keys,
                              std::vector<int> *indices) {
    arrow::FieldVector fields = schema->fields();
    for (const auto &key : keys) {
        std::vector<arrow::FieldPath> field_paths = key.FindAll(fields);
        assert(field_paths.size() == 1 && "Field path ambiguous");
        ARROW_ASSIGN_OR_RAISE(std::shared_ptr<arrow::Field> field, field_paths[0].Get(fields));
        int index = std::find(fields.begin(), fields.end(), field) - fields.begin();
        indices->push_back(index);
    }
    return arrow::Status::OK();
}

std::unique_ptr<::cudf::groupby_aggregation> arrow_to_cudf_aggregation(
    std::string agg, std::shared_ptr<arrow::compute::FunctionOptions> opt = nullptr) {
    if (agg == "hash_approximate_median")
        return ::cudf::make_median_aggregation<::cudf::groupby_aggregation>();
    else if (agg == "hash_count") {
        if (!opt) return ::cudf::make_count_aggregation<::cudf::groupby_aggregation>();
        std::shared_ptr<arrow::compute::CountOptions> copts =
            std::static_pointer_cast<arrow::compute::CountOptions>(opt);
        assert(copts);
        if (copts->mode == arrow::compute::CountOptions::ONLY_VALID)
            return ::cudf::make_count_aggregation<::cudf::groupby_aggregation>(
                ::cudf::null_policy::EXCLUDE);
        else if (copts->mode == arrow::compute::CountOptions::ALL)
            return ::cudf::make_count_aggregation<::cudf::groupby_aggregation>(
                ::cudf::null_policy::INCLUDE);
    }
    if (agg == "hash_sum" || agg == "sum")
        return ::cudf::make_sum_aggregation<::cudf::groupby_aggregation>();
    else if (agg == "hash_min" || agg == "min")
        return ::cudf::make_min_aggregation<::cudf::groupby_aggregation>();
    else if (agg == "hash_max" || agg == "max")
        return ::cudf::make_max_aggregation<::cudf::groupby_aggregation>();
    else if (agg == "hash_mean" || agg == "mean")
        return ::cudf::make_mean_aggregation<::cudf::groupby_aggregation>();
    else if (agg == "hash_count_distinct")
        return ::cudf::make_nunique_aggregation<::cudf::groupby_aggregation>();
    else if (agg == "hash_stddev" || agg == "stddev")
        return ::cudf::make_std_aggregation<::cudf::groupby_aggregation>();
    throw std::runtime_error("The aggregation function not supported in CUDF.");

    return nullptr;
}

GroupByOperator::GroupByOperator(std::shared_ptr<MaximusContext> &_ctx,
                                 std::shared_ptr<Schema> _input_schema,
                                 std::shared_ptr<GroupByProperties> _properties)
        : AbstractGroupByOperator(_ctx, _input_schema, std::move(_properties))
        , GpuOperator(_ctx, {_input_schema}, get_id(), {0}) {
    assert(ctx_);
    auto gctx = ctx_->get_gpu_context();

    assert(gctx != nullptr && "MaximusGContext must be initialized "
                              "before creating a GPU GroupByOperator");

    assert(typeid(*gctx) == typeid(maximus::gpu::MaximusCudaContext) &&
           "MaximusGContext must be a MaximusCudaContext");

    // get the key indices
    key_indices.clear();
    std::shared_ptr<maximus::Schema> input_schema = input_schemas[0];
    assert(input_schema != nullptr && "Input schema is null");
    std::shared_ptr<arrow::Schema> arrow_schema = input_schema->get_schema();
    arrow::FieldVector fields                   = arrow_schema->fields();
    arrow::Status status = get_key_indices(arrow_schema, properties->group_keys, &key_indices);
    assert(status.ok() && "Failed to get key indices");

    // create the output schema and the aggregation list
    arrow::FieldVector out_fields;
    for (int index : key_indices) {
        out_fields.push_back(fields[index]);
    }
    for (auto &aggr : properties->aggregates) {
        arrow::compute::Aggregate arrow_aggregate = *aggr->get_aggregate();
        assert(arrow_aggregate.target.size() == 1 && "Only one target column is expected, for now");
        for (auto &target_col : arrow_aggregate.target) {
            std::vector<arrow::FieldPath> field_paths = target_col.FindAll(fields);
            assert(field_paths.size() == 1 && "Aggregation path ambiguous");
            arrow::Result<std::shared_ptr<arrow::Field>> field_result = field_paths[0].Get(fields);
            assert(field_result.ok() && "Failed to get field");
            std::shared_ptr<arrow::Field> field = field_result.ValueOrDie();
            int index = std::find(fields.begin(), fields.end(), field) - fields.begin();
            std::unique_ptr<::cudf::groupby_aggregation> cudf_aggregation =
                arrow_to_cudf_aggregation(arrow_aggregate.function, arrow_aggregate.options);
            assert(cudf_aggregation != nullptr &&
                   "Failed to convert arrow aggregation to cudf aggregation");
            std::shared_ptr<arrow::DataType> type =
                maximus::gpu::to_arrow_type(::cudf::detail::target_type(
                    maximus::gpu::to_cudf_type(field->type()), cudf_aggregation->kind));
            aggregations.push_back({index, std::move(cudf_aggregation)});
            agg_strings.push_back({index, arrow_aggregate.function});
            std::string name = (arrow_aggregate.name.empty())
                                   ? field->name() + "_" + arrow_aggregate.function
                                   : arrow_aggregate.name;
            out_fields.emplace_back(arrow::field(name, type));
        }
    }
    output_schema = std::make_shared<Schema>(arrow::schema(out_fields));

    set_device_type(DeviceType::GPU);
    set_engine_type(EngineType::CUDF);

    operator_name = name();
}

void GroupByOperator::on_add_input(DeviceTablePtr device_input, int port) {
    proxy_add_input(device_input, port);
}

void GroupByOperator::run_kernel(std::shared_ptr<MaximusContext> &ctx,
                                 std::vector<CudfTablePtr> &input_tables,
                                 std::vector<CudfTablePtr> &output_tables) {
    // there is only one input port
    assert(input_tables.size() == 1);
    assert(input_tables[0]);

    auto &input = input_tables[0];

    std::shared_ptr<::cudf::table> combined_table = std::move(input);
    ::cudf::table_view complete_view              = combined_table->view();

    if (key_indices.empty()) {
        // we need to run reductions on the entire table
        std::vector<std::unique_ptr<::cudf::column>> output_cols;
        std::vector<int> segment = {0, complete_view.num_rows()};
        thrust::device_vector<::cudf::size_type> dev_segment(segment.begin(), segment.end());
        for (auto &aggr : agg_strings) {
            if (aggr.second == "sum") {
                auto agg = ::cudf::make_sum_aggregation<::cudf::segmented_reduce_aggregation>();
                ::cudf::data_type type =
                    ::cudf::detail::target_type(complete_view.column(aggr.first).type(), agg->kind);
                output_cols.push_back(::cudf::segmented_reduce(
                    complete_view.column(aggr.first),
                    ::cudf::device_span<::cudf::size_type const>(dev_segment),
                    *agg,
                    type,
                    ::cudf::null_policy::EXCLUDE));
            } else if (aggr.second == "max") {
                auto agg = ::cudf::make_max_aggregation<::cudf::segmented_reduce_aggregation>();
                ::cudf::data_type type =
                    ::cudf::detail::target_type(complete_view.column(aggr.first).type(), agg->kind);
                output_cols.push_back(::cudf::segmented_reduce(
                    complete_view.column(aggr.first),
                    ::cudf::device_span<::cudf::size_type const>(dev_segment),
                    *agg,
                    type,
                    ::cudf::null_policy::EXCLUDE));
            } else if (aggr.second == "min") {
                auto agg = ::cudf::make_min_aggregation<::cudf::segmented_reduce_aggregation>();
                ::cudf::data_type type =
                    ::cudf::detail::target_type(complete_view.column(aggr.first).type(), agg->kind);
                output_cols.push_back(::cudf::segmented_reduce(
                    complete_view.column(aggr.first),
                    ::cudf::device_span<::cudf::size_type const>(dev_segment),
                    *agg,
                    type,
                    ::cudf::null_policy::EXCLUDE));
            } else if (aggr.second == "mean") {
                auto agg = ::cudf::make_mean_aggregation<::cudf::segmented_reduce_aggregation>();
                ::cudf::data_type type =
                    ::cudf::detail::target_type(complete_view.column(aggr.first).type(), agg->kind);
                output_cols.push_back(::cudf::segmented_reduce(
                    complete_view.column(aggr.first),
                    ::cudf::device_span<::cudf::size_type const>(dev_segment),
                    *agg,
                    type,
                    ::cudf::null_policy::EXCLUDE));
            } else {
                std::__throw_runtime_error("Not implemented yet");
            }
        }

        auto output_table = std::make_shared<::cudf::table>(std::move(output_cols));
        output_tables.emplace_back(std::move(output_table));
        return;
    }

    // create groupby object
    ::cudf::groupby::groupby groupby_obj(complete_view.select(key_indices));

    // create aggregation requests
    std::vector<::cudf::groupby::aggregation_request> requests;
    for (auto &aggr : aggregations) {
        std::vector<std::unique_ptr<::cudf::groupby_aggregation>> aggregation_fns;
        aggregation_fns.push_back(std::move(aggr.second));
        requests.push_back({complete_view.column(aggr.first), std::move(aggregation_fns)});
    }

    // perform groupby
    std::pair<std::unique_ptr<::cudf::table>, std::vector<::cudf::groupby::aggregation_result>>
        result = groupby_obj.aggregate(
            ::cudf::host_span<::cudf::groupby::aggregation_request const>(requests));

    // export the result to a GTable
    std::vector<std::unique_ptr<::cudf::column>> output_cols = result.first->release();
    for (auto &agg_result : result.second) {
        assert(agg_result.results.size() == 1 && "Only one output column is expected");
        output_cols.push_back(std::move(agg_result.results[0]));
    }

    // std::cout << (int) output_cols[1]->type().id() << std::endl;
    auto output_table = std::make_shared<::cudf::table>(std::move(output_cols));
    output_tables.emplace_back(std::move(output_table));
}

void GroupByOperator::on_no_more_input(int port) {
    proxy_no_more_input(port);
}

bool GroupByOperator::has_more_batches_impl(bool blocking) {
    return proxy_has_more_batches(blocking);
}

DeviceTablePtr GroupByOperator::export_next_batch_impl() {
    return std::move(proxy_export_next_batch());
}

}  // namespace maximus::cudf
