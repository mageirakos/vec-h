#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/sorting.hpp>
#include <maximus/gpu/cuda_api.hpp>
#include <maximus/operators/gpu/cudf/hash_join_operator.hpp>
#include <typeinfo>

namespace maximus::cudf {

arrow::Status get_hash_indices(const std::shared_ptr<arrow::Schema> &schema,
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

HashJoinOperator::HashJoinOperator(std::shared_ptr<MaximusContext> &_ctx,
                                   std::shared_ptr<Schema> _left_schema,
                                   std::shared_ptr<Schema> _right_schema,
                                   std::shared_ptr<JoinProperties> _properties)
        : AbstractHashJoinOperator(_ctx, _left_schema, _right_schema, std::move(_properties))
        , GpuOperator(_ctx, {_left_schema, _right_schema}, get_id(), {0, 1}) {
    assert(ctx_);
    auto gctx = ctx_->get_gpu_context();

    assert(gctx != nullptr && "MaximusGContext must be initialized "
                              "before creating a GPU HashJoinOperator");

    assert(typeid(*gctx) == typeid(maximus::gpu::MaximusCudaContext) &&
           "MaximusGContext must be a MaximusCudaContext");

    // get key indices for build table
    build_key_indices.clear();
    std::shared_ptr<maximus::Schema> build_schema = input_schemas[0];
    assert(build_schema != nullptr && "Build schema is null");
    std::shared_ptr<arrow::Schema> arrow_schema = build_schema->get_schema();
    arrow::FieldVector fields                   = arrow_schema->fields();
    arrow::Status status =
        get_hash_indices(arrow_schema, properties->left_keys, &build_key_indices);
    assert(status.ok() && "Failed to get build key indices");

    // get key indices for probe table
    probe_key_indices.clear();
    std::shared_ptr<maximus::Schema> probe_schema = input_schemas[1];
    assert(probe_schema != nullptr && "Probe schema is null");
    arrow_schema = probe_schema->get_schema();
    fields       = arrow_schema->fields();
    status       = get_hash_indices(arrow_schema, properties->right_keys, &probe_key_indices);
    assert(status.ok() && "Failed to get probe key indices");

    // get suffixes and join type
    build_suffix = properties->left_suffix;
    probe_suffix = properties->right_suffix;
    join_type    = properties->join_type;

    // expressions - not supported for now

    // create the output schema
    if (join_type == JoinType::LEFT_ANTI || join_type == JoinType::LEFT_SEMI) {
        // output schema is the same as the build schema
        output_schema = build_schema;
    } else if (join_type == JoinType::RIGHT_ANTI || join_type == JoinType::RIGHT_SEMI) {
        // output schema is the same as the probe schema
        output_schema = probe_schema;
    } else {
        // output schema is a combination of both schemas
        arrow::FieldVector out_fields;
        std::unordered_map<std::string, int32_t> column_name_index;

        // adding build table
        for (const auto &field : build_schema->get_schema()->fields()) {
            column_name_index.insert(std::make_pair<>(field->name(), out_fields.size()));
            out_fields.emplace_back(field);
        }

        // adding probe table
        for (const auto &field : probe_schema->get_schema()->fields()) {
            auto new_field = field;
            if (column_name_index.find(field->name()) != column_name_index.end()) {
                assert((build_suffix != probe_suffix) && "Suffixes must be different");
                out_fields[column_name_index.find(field->name())->second] =
                    field->WithName(field->name() + build_suffix);
                new_field = field->WithName(field->name() + probe_suffix);
            }
            column_name_index.insert(std::make_pair<>(new_field->name(), out_fields.size()));
            out_fields.emplace_back(new_field);
        }

        output_schema = std::make_shared<Schema>(std::make_shared<arrow::Schema>(out_fields));
    }

    // in cudf, both tables must be fully received before processing
    // so both input ports are blocking
    AbstractOperator::set_blocking_port(0);
    AbstractOperator::set_blocking_port(1);

    set_device_type(DeviceType::GPU);
    set_engine_type(EngineType::CUDF);

    operator_name = name();
}

int HashJoinOperator::get_build_port() const {
    return -1;
}

int HashJoinOperator::get_probe_port() const {
    return -1;
}

void HashJoinOperator::on_add_input(DeviceTablePtr device_input, int port) {
    proxy_add_input(device_input, port);
}

std::vector<std::unique_ptr<::cudf::column>> gather_column(
    ::cudf::table_view const &input,
    rmm::device_uvector<::cudf::size_type> join_indices,
    ::cudf::out_of_bounds_policy oob_policy = ::cudf::out_of_bounds_policy::DONT_CHECK) {
    ::cudf::device_span<::cudf::size_type const> indices_span =
        ::cudf::device_span<::cudf::size_type const>{join_indices};
    ::cudf::column_view indices_col            = ::cudf::column_view{indices_span};
    std::unique_ptr<::cudf::table> left_result = ::cudf::gather(input, indices_col, oob_policy);
    std::vector<std::unique_ptr<::cudf::column>> joined_cols = left_result->release();
    return std::move(joined_cols);
}

template<auto join_impl,
         ::cudf::out_of_bounds_policy oob_policy = ::cudf::out_of_bounds_policy::DONT_CHECK>
std::shared_ptr<::cudf::table> join_and_gather_left(
    ::cudf::table_view const &left_input,
    ::cudf::table_view const &right_input,
    std::vector<::cudf::size_type> const &left_key_indices,
    std::vector<::cudf::size_type> const &right_key_indices,
    ::cudf::null_equality compare_nulls) {
    auto const [left_join_indices, right_join_indices] =
        (*join_impl)(left_input.select(left_key_indices),
                     right_input.select(right_key_indices),
                     compare_nulls,
                     ::cudf::get_default_stream(),
                     rmm::mr::get_current_device_resource());

    std::vector<std::unique_ptr<::cudf::column>>
        joined_cols = gather_column(left_input, std::move(*left_join_indices), oob_policy),
        right_cols  = gather_column(right_input, std::move(*right_join_indices), oob_policy);

    joined_cols.insert(joined_cols.end(),
                       std::make_move_iterator(right_cols.begin()),
                       std::make_move_iterator(right_cols.end()));
    return std::make_shared<::cudf::table>(std::move(joined_cols));
}

template<auto join_impl,
         ::cudf::out_of_bounds_policy oob_policy = ::cudf::out_of_bounds_policy::DONT_CHECK>
std::shared_ptr<::cudf::table> join_and_gather_right(
    ::cudf::table_view const &left_input,
    ::cudf::table_view const &right_input,
    std::vector<::cudf::size_type> const &left_key_indices,
    std::vector<::cudf::size_type> const &right_key_indices,
    ::cudf::null_equality compare_nulls) {
    auto const [right_join_indices, left_join_indices] =
        (*join_impl)(right_input.select(right_key_indices),
                     left_input.select(left_key_indices),
                     compare_nulls,
                     ::cudf::get_default_stream(),
                     rmm::mr::get_current_device_resource());

    std::vector<std::unique_ptr<::cudf::column>>
        joined_cols = gather_column(left_input, std::move(*left_join_indices), oob_policy),
        right_cols  = gather_column(right_input, std::move(*right_join_indices), oob_policy);

    joined_cols.insert(joined_cols.end(),
                       std::make_move_iterator(right_cols.begin()),
                       std::make_move_iterator(right_cols.end()));
    return std::make_shared<::cudf::table>(std::move(joined_cols));
}

template<auto join_impl,
         ::cudf::out_of_bounds_policy oob_policy = ::cudf::out_of_bounds_policy::DONT_CHECK>
std::shared_ptr<::cudf::table> semi_join_and_gather_left(
    ::cudf::table_view const &left_input,
    ::cudf::table_view const &right_input,
    std::vector<::cudf::size_type> const &left_key_indices,
    std::vector<::cudf::size_type> const &right_key_indices,
    ::cudf::null_equality compare_nulls) {
    auto const left_join_indices = (*join_impl)(left_input.select(left_key_indices),
                                                right_input.select(right_key_indices),
                                                compare_nulls,
                                                ::cudf::get_default_stream(),
                                                rmm::mr::get_current_device_resource());
    return std::make_shared<::cudf::table>(
        gather_column(left_input, std::move(*left_join_indices), oob_policy));
}

template<auto join_impl,
         ::cudf::out_of_bounds_policy oob_policy = ::cudf::out_of_bounds_policy::DONT_CHECK>
std::shared_ptr<::cudf::table> semi_join_and_gather_right(
    ::cudf::table_view const &left_input,
    ::cudf::table_view const &right_input,
    std::vector<::cudf::size_type> const &left_key_indices,
    std::vector<::cudf::size_type> const &right_key_indices,
    ::cudf::null_equality compare_nulls) {
    auto const right_join_indices = (*join_impl)(right_input.select(right_key_indices),
                                                 left_input.select(left_key_indices),
                                                 compare_nulls,
                                                 ::cudf::get_default_stream(),
                                                 rmm::mr::get_current_device_resource());

    return std::make_shared<::cudf::table>(
        gather_column(right_input, std::move(*right_join_indices), oob_policy));
}

void HashJoinOperator::on_no_more_input(int port) {
    proxy_no_more_input(port);
}

bool HashJoinOperator::handle_empty_inputs(std::vector<CudfTablePtr> &input_tables) {
    // For outer/full joins, the kernel must still run when one side is empty
    // (cudf handles this correctly and emits the non-empty side with NULLs).
    // Replace any nullptr entries with schema-correct empty tables so run_kernel can proceed.
    bool needs_kernel = (join_type == JoinType::LEFT_OUTER ||
                         join_type == JoinType::RIGHT_OUTER ||
                         join_type == JoinType::FULL_OUTER);
    if (!needs_kernel) {
        return false;
    }
    for (int i = 0; i < input_tables.size(); i++) {
        if (!input_tables[i]) {
            input_tables[i] = maximus::gpu::make_empty_cudf_table(_input_schemas[i]);
        }
    }
    return true;
}

void HashJoinOperator::run_kernel(std::shared_ptr<MaximusContext> &ctx,
                                  std::vector<CudfTablePtr> &input_tables,
                                  std::vector<CudfTablePtr> &output_tables) {
    // there are only 2 input ports
    assert(input_tables.size() == 2);
    assert(input_tables[0]);
    assert(input_tables[1]);

    auto &build_combined_table = input_tables[0];
    auto &probe_combined_table = input_tables[1];

    auto build_complete_view = build_combined_table->view();
    auto probe_complete_view = probe_combined_table->view();

    // build the hash join
    std::shared_ptr<::cudf::table> result;
    switch (join_type) {
        case JoinType::LEFT_SEMI:
            result = std::move(
                semi_join_and_gather_left<&::cudf::left_semi_join>(build_complete_view,
                                                                   probe_complete_view,
                                                                   build_key_indices,
                                                                   probe_key_indices,
                                                                   ::cudf::null_equality::EQUAL));
            break;
        case JoinType::RIGHT_SEMI:
            result = std::move(
                semi_join_and_gather_right<&::cudf::left_semi_join>(build_complete_view,
                                                                    probe_complete_view,
                                                                    build_key_indices,
                                                                    probe_key_indices,
                                                                    ::cudf::null_equality::EQUAL));
            break;
        case JoinType::LEFT_ANTI:
            result = std::move(
                semi_join_and_gather_left<&::cudf::left_anti_join>(build_complete_view,
                                                                   probe_complete_view,
                                                                   build_key_indices,
                                                                   probe_key_indices,
                                                                   ::cudf::null_equality::EQUAL));
            break;
        case JoinType::RIGHT_ANTI:
            result = std::move(
                semi_join_and_gather_right<&::cudf::left_anti_join>(build_complete_view,
                                                                    probe_complete_view,
                                                                    build_key_indices,
                                                                    probe_key_indices,
                                                                    ::cudf::null_equality::EQUAL));
            break;
        case JoinType::INNER:
            result =
                std::move(join_and_gather_left<&::cudf::inner_join>(build_complete_view,
                                                                    probe_complete_view,
                                                                    build_key_indices,
                                                                    probe_key_indices,
                                                                    ::cudf::null_equality::EQUAL));
            break;
        case JoinType::LEFT_OUTER:
            result = std::move(
                join_and_gather_left<&::cudf::left_join, ::cudf::out_of_bounds_policy::NULLIFY>(
                    build_complete_view,
                    probe_complete_view,
                    build_key_indices,
                    probe_key_indices,
                    ::cudf::null_equality::EQUAL));
            break;
        case JoinType::RIGHT_OUTER:
            result = std::move(
                join_and_gather_right<&::cudf::left_join, ::cudf::out_of_bounds_policy::NULLIFY>(
                    build_complete_view,
                    probe_complete_view,
                    build_key_indices,
                    probe_key_indices,
                    ::cudf::null_equality::EQUAL));
            break;
        case JoinType::FULL_OUTER:
            result = std::move(
                join_and_gather_left<&::cudf::full_join, ::cudf::out_of_bounds_policy::NULLIFY>(
                    build_complete_view,
                    probe_complete_view,
                    build_key_indices,
                    probe_key_indices,
                    ::cudf::null_equality::EQUAL));
            break;
        case JoinType::CROSS_JOIN:
            result = std::move(::cudf::cross_join(build_complete_view, probe_complete_view));
            break;
    }

    // auto result_sorted =
    //     ::cudf::sort(result->view(),
    //                  std::vector<::cudf::order>(result->num_columns(), ::cudf::order::ASCENDING));

    // export the result to a GTable
    output_tables.emplace_back(std::move(result));
}

bool HashJoinOperator::has_more_batches_impl(bool blocking) {
    return proxy_has_more_batches(blocking);
}

DeviceTablePtr HashJoinOperator::export_next_batch_impl() {
    return std::move(proxy_export_next_batch());
}

}  // namespace maximus::cudf
