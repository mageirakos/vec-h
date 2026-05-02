#include "queries.hpp"

#include <arrow/compute/api.h>
#include <arrow/compute/expression.h>
#include <faiss/IndexHNSW.h>

#include <maximus/frontend/expressions.hpp>
#include <maximus/frontend/query_plan_api.hpp>
#include <maximus/indexes/index.hpp>
#include <maximus/operators/faiss/join_operator.hpp>
#include <maximus/types/expression.hpp>
#include <maximus/utils/arrow_helpers.hpp>
#include <maximus/utils/utils.hpp>
#include <regex>

#include "utils.hpp"

#if defined(MAXIMUS_WITH_CUDA) && defined(MAXIMUS_WITH_FAISS_GPUCUVS)
#include <faiss/gpu/GpuIndex.h>
#include <faiss/gpu/GpuIndexCagra.h>
#endif

using SchemaPtr      = std::shared_ptr<maximus::Schema>;
using ExpressionPtr  = std::shared_ptr<maximus::Expression>;
using Database       = maximus::Database;
using QualityMetrics = maximus::QualityMetrics;
using QueryPlan      = maximus::QueryPlan;
using QueryNode      = maximus::QueryNode;
using Schema         = maximus::Schema;

namespace big_vector_bench {

/************************************************************************************/
//                                   Workload
/************************************************************************************/

class Workload : public AbstractWorkload {
public:
    Workload(std::shared_ptr<Database>& db): db(db) {}
    static QualityMetrics evaluate_retrieval_quality(std::vector<maximus::TablePtr> results,
                                                     std::shared_ptr<Database>& db,
                                                     const std::string& gt_table) {
        const auto& table = db->get_table_nocopy(gt_table).as_table()->get_table();
        // Ground truth is in FixedSizeList Format
        auto chunked_array = table->GetColumnByName("neighbors");
        if (chunked_array->num_chunks() != 1) {
            throw std::runtime_error("Expected exactly 1 chunk");
        }
        // get the single chunk
        auto array_chunk = chunked_array->chunk(0);

        // Get schema field type for this column
        auto field_type = table->schema()->GetFieldByName("neighbors")->type();

        // Determine the embedding dimension
        int list_size = maximus::embedding_dimension(array_chunk);

        // get the actual embeddings data as a single, flattened array
        auto values_array = maximus::embeddings_values(array_chunk);

        int total_vectors_gt = list_size * array_chunk->length();

        int total_vectors_retrieved = 0;
        int total_tp                = 0;
        for (int i = 0; i < results.size(); i++) {
            if (results[i] == nullptr) {
                std::cout << "Query " << i << " result is nullptr (empty?)" << std::endl;
                continue;
            } else if (results[i]->num_rows() < 100) {
                std::cout << "Query " << i << ": " << results[i]->num_rows() << std::endl;
                continue;
            }
            auto gt_ids = std::dynamic_pointer_cast<arrow::Int32Array>(
                values_array->Slice(i * list_size, list_size));
            auto result_ids = results[i]->get_table()->GetColumnByName("id");
            std::shared_ptr<arrow::Array> concatenated_ids =
                arrow::Concatenate(result_ids->chunks(), arrow::default_memory_pool()).ValueOrDie();
            auto casted = std::dynamic_pointer_cast<arrow::Int32Array>(concatenated_ids);
            //std::cout << "Evaluating query " << i << "/" << results.size() << std::endl;
            total_tp += maximus::compute_intersection_size(casted, gt_ids);
            total_vectors_retrieved += results[i]->num_rows();
        }
        double recall    = total_tp / (double) total_vectors_gt;
        double precision = total_tp / (double) total_vectors_retrieved;
        return {recall, precision, total_vectors_retrieved};
    }

    static std::shared_ptr<maximus::faiss::FaissSearchParameters> make_search_parameters(
        const QueryParameters& query_parameters) {
        if (maximus::starts_with(query_parameters.faiss_index, "HNSW")) {
            auto parameters      = std::make_unique<::faiss::SearchParametersHNSW>();
            parameters->efSearch = query_parameters.hnsw_efsearch;
            auto parameters_wrapped =
                std::make_shared<maximus::faiss::FaissSearchParameters>(std::move(parameters));
            std::cout << "Using search_parameters with efsearch=" << query_parameters.hnsw_efsearch
                      << std::endl;
            return parameters_wrapped;
        } else if (maximus::starts_with(query_parameters.faiss_index, "IVF")) {
            auto parameters    = std::make_unique<::faiss::SearchParametersIVF>();
            parameters->nprobe = query_parameters.ivf_nprobe;
            auto parameters_wrapped =
                std::make_shared<maximus::faiss::FaissSearchParameters>(std::move(parameters));
            std::cout << "Using search_parameters with nprobe=" << query_parameters.ivf_nprobe
                      << std::endl;
            return parameters_wrapped;
        }
#if defined(MAXIMUS_WITH_CUDA) && defined(MAXIMUS_WITH_FAISS_GPUCUVS)
        else if (maximus::starts_with(query_parameters.faiss_index, "GPU,Cagra")) {
            auto parameters          = std::make_unique<::faiss::gpu::SearchParametersCagra>();
            parameters->itopk_size   = query_parameters.cagra_itopksize;
            parameters->search_width = query_parameters.cagra_searchwidth;
            auto parameters_wrapped =
                std::make_shared<maximus::faiss::FaissSearchParameters>(std::move(parameters));
            std::cout << "Using search_parameters with itopksize="
                      << query_parameters.cagra_itopksize
                      << " and searchwidth=" << query_parameters.cagra_searchwidth << std::endl;
            return parameters_wrapped;
        }
#endif
        else {
            std::cout << "Not using any search parameters." << std::endl;
            return nullptr;
        }
    }

    static maximus::faiss::FaissIndexPtr make_index(std::shared_ptr<Database>& db,
                                                    const std::string& train_table,
                                                    const std::string& train_vector_column,
                                                    const QueryParameters& query_parameters) {
        // Extract data
        auto table = db->get_table_nocopy(train_table);
        table.convert_to<maximus::TableBatchPtr>(db->get_context(), table.as_table()->get_schema());
        const auto& table_batch = table.as_table_batch();
        auto column     = table_batch->get_table_batch()->GetColumnByName(train_vector_column);
        auto field_type = table_batch->get_schema()->column_types()[train_vector_column];

        const int d = maximus::embedding_dimension(column);

        // Build index
        auto vector_index = maximus::faiss::FaissIndex::factory_make(
            db->get_context(), d, query_parameters.faiss_index);

        PE("Faiss Index Creation");
        // Cast and call the appropriate overload depending on the Arrow type
        switch (column->type()->id()) {
            case arrow::Type::FIXED_SIZE_LIST: {
                auto vectors_fixed = std::static_pointer_cast<arrow::FixedSizeListArray>(column);
                vector_index->train(*vectors_fixed);
                vector_index->add(*vectors_fixed);
                break;
            }
            case arrow::Type::LIST: {
                auto vectors_list = std::static_pointer_cast<maximus::EmbeddingsArray>(column);
                vector_index->train(*vectors_list);
                vector_index->add(*vectors_list);
                break;
            }
            default:
                throw std::runtime_error("Unsupported vector column type for index training: " +
                                         column->type()->ToString());
        }
        PL("Faiss Index Creation");
        std::cout << "Index built" << std::endl;
        return vector_index;
    }

public:
    std::shared_ptr<Database> db;
};

/************************************************************************************/
//                                  Filter Workload
/************************************************************************************/

class FilterWorkload : public Workload {
public:
    FilterWorkload(std::shared_ptr<Database>& db): Workload(db){};
    virtual ExpressionPtr get_filter_expression() const                                       = 0;
    virtual ExpressionPtr get_filter_expression_semibound(std::vector<int> test_labels) const = 0;

    std::vector<std::shared_ptr<maximus::QueryPlan>> postFiltering(
        std::shared_ptr<Database>& db,
        const std::string& test_table,
        const std::string& train_table,
        SchemaPtr test_table_schema,
        SchemaPtr train_table_schema,
        const int num_test_rows,
        const QueryParameters& query_parameters) const {
        const std::string train_vector_column = "train_vec";  // Holds for all filter-ann datasets
        const std::string test_vector_column  = "test_vec";   // Holds for all filter-ann datasets
        const int K                           = 100;          // Holds for all filter-ann datasets

        auto filter_expr = get_filter_expression();
        auto ctx         = db->get_context();

        // Build Index
        auto vector_index = make_index(db, train_table, train_vector_column, query_parameters);
        auto parameters   = make_search_parameters(query_parameters);

        std::vector<std::shared_ptr<QueryPlan>> queries;
        queries.reserve(num_test_rows);
        for (int i = 0; i < num_test_rows; i++) {
            // DAG
            auto data_source_node  = table_source(db,
                                                 train_table,
                                                 train_table_schema,
                                                 train_table_schema->column_names(),
                                                 maximus::DeviceType::CPU,
                                                 true);  // nocopy_variant=true
            auto query_source_node = table_source(db,
                                                  test_table,
                                                  test_table_schema,
                                                  test_table_schema->column_names(),
                                                  maximus::DeviceType::CPU,
                                                  true);

            auto rename_node = rename(query_source_node,
                                      {"id", test_vector_column, "test_label"},
                                      {"query_id", test_vector_column, "test_label"});
            // Limit
            auto limit_node =
                std::make_shared<QueryNode>(maximus::EngineType::NATIVE,
                                            maximus::DeviceType::CPU,
                                            maximus::NodeType::LIMIT,
                                            std::make_shared<maximus::LimitProperties>(1, i),
                                            ctx);
            limit_node->add_input(rename_node);

            auto join_properties = std::make_shared<maximus::VectorJoinIndexedProperties>(
                arrow::FieldRef(train_vector_column),
                arrow::FieldRef(test_vector_column),
                vector_index,
                query_parameters.postfilter_ksearch,
                std::nullopt,
                parameters,
                false,
                false,
                "distance");
            auto join_node = std::make_shared<QueryNode>(maximus::EngineType::FAISS,
                                                         maximus::DeviceType::CPU,
                                                         maximus::NodeType::VECTOR_JOIN,
                                                         std::move(join_properties),
                                                         ctx);
            join_node->add_input(data_source_node);
            join_node->add_input(limit_node);

            auto filter_node = filter(join_node, filter_expr, maximus::DeviceType::CPU);

            auto final_order_by_node =
                order_by(filter_node,
                         {maximus::SortKey("distance", maximus::SortOrder::ASCENDING)},
                         maximus::DeviceType::CPU);
            auto final_limit_node = limit(final_order_by_node, K, 0);

            queries.push_back(query_plan(table_sink(final_limit_node)));
        }
        return queries;
    }


    std::vector<std::shared_ptr<maximus::QueryPlan>> postFilteringWithExpression(
        std::shared_ptr<Database>& db,
        const std::string& test_table,
        const std::string& train_table,
        SchemaPtr test_table_schema,
        SchemaPtr train_table_schema,
        const int num_test_rows,
        const QueryParameters& query_parameters) const {
        const std::string train_vector_column = "train_vec";  // Holds for all filter-ann datasets
        const std::string test_vector_column  = "test_vec";   // Holds for all filter-ann datasets
        const int K                           = 100;          // Holds for all filter-ann datasets
        const int label_column_offset         = 2;            // Holds for all filter-ann datasets
        const int num_ground_truth_columns    = 2;            // Holds for all filter-ann datasets

        auto filter_expr     = get_filter_expression();
        auto ctx             = db->get_context();
        auto test_table_data = db->get_table_nocopy(test_table);
        auto test_columns    = test_table_schema->column_names();
        auto test_label_columns =
            std::vector<std::string>(test_columns.begin() + label_column_offset,
                                     test_columns.end() - num_ground_truth_columns);

        // Build Index
        auto vector_index = make_index(db, train_table, train_vector_column, query_parameters);
        std::cout << "Index built" << std::endl;
        auto parameters = make_search_parameters(query_parameters);


        std::vector<std::shared_ptr<QueryPlan>> queries;
        queries.reserve(num_test_rows);
        for (int i = 0; i < num_test_rows; i++) {
            // DAG
            auto data_source_node =
                table_source(db,
                             train_table,
                             train_table_schema,
                             train_table_schema->column_names(),
                             maximus::DeviceType::CPU,
                             true);  // nocopy_variant=true since we're looping over test rows
            auto query_source_node = table_source(db,
                                                  test_table,
                                                  test_table_schema,
                                                  test_table_schema->column_names(),
                                                  maximus::DeviceType::CPU,
                                                  true);

            auto rename_node = rename(query_source_node,
                                      {"id", test_vector_column, "test_label"},
                                      {"query_id", test_vector_column, "test_label"});
            // Limit
            auto limit_node =
                std::make_shared<QueryNode>(maximus::EngineType::NATIVE,
                                            maximus::DeviceType::CPU,
                                            maximus::NodeType::LIMIT,
                                            std::make_shared<maximus::LimitProperties>(1, i),
                                            ctx);
            limit_node->add_input(rename_node);

            // Filter prep
            std::vector<int> test_label_values;
            for (auto& column_name : test_label_columns) {
                auto column = test_table_data.as_table()->get_table()->GetColumnByName(column_name);
                auto label_scalar = column->GetScalar(i);
                CHECK_STATUS(label_scalar.status());
                auto label_scalar_int =
                    std::dynamic_pointer_cast<arrow::Int32Scalar>(label_scalar.ValueOrDie());
                int value = label_scalar_int->value;
                test_label_values.push_back(value);
            }

            // Join
            auto join_properties = std::make_shared<maximus::VectorJoinIndexedProperties>(
                arrow::FieldRef(train_vector_column),
                arrow::FieldRef(test_vector_column),
                vector_index,
                K,
                std::nullopt,
                parameters,
                false,
                false,
                "distance",
                get_filter_expression_semibound(test_label_values));
            auto join_node = std::make_shared<QueryNode>(maximus::EngineType::FAISS,
                                                         maximus::DeviceType::CPU,
                                                         maximus::NodeType::VECTOR_JOIN,
                                                         std::move(join_properties),
                                                         ctx);
            join_node->add_input(data_source_node);
            join_node->add_input(limit_node);

            queries.push_back(query_plan(table_sink(join_node)));
        }
        return queries;
    }


    std::vector<std::shared_ptr<maximus::QueryPlan>> preFilteringWithProjectAndIndex(
        std::shared_ptr<Database>& db,
        const std::string& test_table,
        const std::string& train_table,
        SchemaPtr test_table_schema,
        SchemaPtr train_table_schema,
        const int num_test_rows,
        const QueryParameters& query_parameters) const {
        const std::string train_vector_column = "train_vec";  // Holds for all filter-ann datasets
        const std::string test_vector_column  = "test_vec";   // Holds for all filter-ann datasets
        const int K                           = 100;          // Holds for all filter-ann datasets
        const int label_column_offset         = 2;            // Holds for all filter-ann datasets
        const int num_ground_truth_columns    = 2;            // Holds for all filter-ann datasets

        auto filter_expr     = get_filter_expression();
        auto ctx             = db->get_context();
        auto test_table_data = db->get_table_nocopy(test_table);
        auto test_columns    = test_table_schema->column_names();
        auto test_label_columns =
            std::vector<std::string>(test_columns.begin() + label_column_offset,
                                     test_columns.end() - num_ground_truth_columns);

        // Build Index
        auto vector_index = make_index(db, train_table, train_vector_column, query_parameters);
        std::cout << "Index built" << std::endl;
        auto parameters = make_search_parameters(query_parameters);

        std::vector<std::shared_ptr<QueryPlan>> queries;
        queries.reserve(num_test_rows);
        for (int i = 0; i < num_test_rows; i++) {
            // DAG
            auto data_source_node  = table_source(db,
                                                 train_table,
                                                 train_table_schema,
                                                 train_table_schema->column_names(),
                                                 maximus::DeviceType::CPU,
                                                 true);
            auto query_source_node = table_source(db,
                                                  test_table,
                                                  test_table_schema,
                                                  test_table_schema->column_names(),
                                                  maximus::DeviceType::CPU,
                                                  true);

            auto rename_node = rename(query_source_node,
                                      {"id", test_vector_column, "test_label"},
                                      {"query_id", test_vector_column, "test_label"});
            // Limit
            auto limit_node =
                std::make_shared<QueryNode>(maximus::EngineType::NATIVE,
                                            maximus::DeviceType::CPU,
                                            maximus::NodeType::LIMIT,
                                            std::make_shared<maximus::LimitProperties>(1, i),
                                            ctx);
            limit_node->add_input(rename_node);

            // FilterProject
            std::vector<int> test_label_values;
            for (auto& column_name : test_label_columns) {
                auto column = test_table_data.as_table()->get_table()->GetColumnByName(column_name);
                auto label_scalar = column->GetScalar(i);
                CHECK_STATUS(label_scalar.status());
                auto label_scalar_int =
                    std::dynamic_pointer_cast<arrow::Int32Scalar>(label_scalar.ValueOrDie());
                int value = label_scalar_int->value;
                test_label_values.push_back(value);
            }
            auto post_project_columns = train_table_schema->column_names();

            // TODO CHANGE: it in the future, if we feel we need to optimize not having 2 copies of data vetors (in arrow/cudf & in FAISS)
            // NOTE: Right now we DO NOT remove the train_vector_column from input to VectorJoin nodes.
            //  - We avoid removing the train_vector_column here to ensure the index can still be used
            //  - AbstractOperators check that the "train_vector_column" exists in the input schema...
            //  - Indeed it is a valid optimization to not "carry" it with us, when FAISS owns the data vectors
            // post_project_columns      = remove_value(post_project_columns, train_vector_column);
            auto project_exprs = maximus::exprs(post_project_columns);
            post_project_columns.push_back("filter");
            project_exprs.push_back(get_filter_expression_semibound(test_label_values));
            auto project_node = project(
                data_source_node, project_exprs, post_project_columns, maximus::DeviceType::CPU);

            // Join
            auto join_properties = std::make_shared<maximus::VectorJoinIndexedProperties>(
                arrow::FieldRef(train_vector_column),
                arrow::FieldRef(test_vector_column),
                vector_index,
                K,
                std::nullopt,
                parameters,
                false,
                false,
                "distance",
                nullptr,
                arrow::FieldRef("filter"));
            auto join_node = std::make_shared<QueryNode>(maximus::EngineType::FAISS,
                                                         maximus::DeviceType::CPU,
                                                         maximus::NodeType::VECTOR_JOIN,
                                                         std::move(join_properties),
                                                         ctx);
            join_node->add_input(project_node);
            join_node->add_input(limit_node);

            queries.push_back(query_plan(table_sink(join_node)));
        }
        return queries;
    }


    std::vector<std::shared_ptr<maximus::QueryPlan>> preFilteringWithProject(
        std::shared_ptr<Database>& db,
        const std::string& test_table,
        const std::string& train_table,
        SchemaPtr test_table_schema,
        SchemaPtr train_table_schema,
        const int num_test_rows,
        const QueryParameters& query_parameters) const {
        const std::string train_vector_column = "train_vec";  // Holds for all filter-ann datasets
        const std::string test_vector_column  = "test_vec";   // Holds for all filter-ann datasets
        const int K                           = 100;          // Holds for all filter-ann datasets
        const int label_column_offset         = 2;            // Holds for all filter-ann datasets
        const int num_ground_truth_columns    = 2;            // Holds for all filter-ann datasets

        auto filter_expr     = get_filter_expression();
        auto ctx             = db->get_context();
        auto test_table_data = db->get_table_nocopy(test_table);
        auto test_columns    = test_table_schema->column_names();
        auto test_label_columns =
            std::vector<std::string>(test_columns.begin() + label_column_offset,
                                     test_columns.end() - num_ground_truth_columns);

        std::vector<std::shared_ptr<QueryPlan>> queries;
        queries.reserve(num_test_rows);
        for (int i = 0; i < num_test_rows; i++) {
            // DAG
            auto data_source_node  = table_source(db,
                                                 train_table,
                                                 train_table_schema,
                                                 train_table_schema->column_names(),
                                                 maximus::DeviceType::CPU,
                                                 true);
            auto query_source_node = table_source(db,
                                                  test_table,
                                                  test_table_schema,
                                                  test_table_schema->column_names(),
                                                  maximus::DeviceType::CPU,
                                                  true);

            auto rename_node = rename(query_source_node,
                                      {"id", test_vector_column, "test_label"},
                                      {"query_id", test_vector_column, "test_label"});
            // Limit
            auto limit_node =
                std::make_shared<QueryNode>(maximus::EngineType::NATIVE,
                                            maximus::DeviceType::CPU,
                                            maximus::NodeType::LIMIT,
                                            std::make_shared<maximus::LimitProperties>(1, i),
                                            ctx);
            limit_node->add_input(rename_node);

            // FilterProject
            std::vector<int> test_label_values;
            for (auto& column_name : test_label_columns) {
                auto column = test_table_data.as_table()->get_table()->GetColumnByName(column_name);
                auto label_scalar = column->GetScalar(i);
                CHECK_STATUS(label_scalar.status());
                auto label_scalar_int =
                    std::dynamic_pointer_cast<arrow::Int32Scalar>(label_scalar.ValueOrDie());
                int value = label_scalar_int->value;
                test_label_values.push_back(value);
            }
            auto post_project_columns = train_table_schema->column_names();
            auto project_exprs        = maximus::exprs(post_project_columns);
            post_project_columns.push_back("filter");
            project_exprs.push_back(get_filter_expression_semibound(test_label_values));
            auto project_node = project(
                data_source_node, project_exprs, post_project_columns, maximus::DeviceType::CPU);

            // Join
            auto join_properties = std::make_shared<maximus::VectorJoinExhaustiveProperties>(
                arrow::FieldRef(train_vector_column),
                arrow::FieldRef(test_vector_column),
                K,
                std::nullopt,
                maximus::VectorDistanceMetric::L2,
                false,
                false,
                "distance",
                arrow::FieldRef("filter"));
            auto join_node = std::make_shared<QueryNode>(maximus::EngineType::FAISS,
                                                         maximus::DeviceType::CPU,
                                                         maximus::NodeType::VECTOR_JOIN,
                                                         std::move(join_properties),
                                                         ctx);
            join_node->add_input(project_node);
            join_node->add_input(limit_node);

            queries.push_back(query_plan(table_sink(join_node)));
        }
        return queries;
    }


    std::vector<std::shared_ptr<maximus::QueryPlan>> preFilteringWithFilter(
        std::shared_ptr<Database>& db,
        const std::string& test_table,
        const std::string& train_table,
        SchemaPtr test_table_schema,
        SchemaPtr train_table_schema,
        const int num_test_rows) const {
        const std::string train_vector_column = "train_vec";  // Holds for all filter-ann datasets
        const std::string test_vector_column  = "test_vec";   // Holds for all filter-ann datasets
        const int K                           = 100;          // Holds for all filter-ann datasets
        const int label_column_offset         = 2;            // Holds for all filter-ann datasets
        const int num_ground_truth_columns    = 2;            // Holds for all filter-ann datasets

        auto ctx             = db->get_context();
        auto test_table_data = db->get_table_nocopy(test_table);
        auto test_columns    = test_table_schema->column_names();
        auto test_label_columns =
            std::vector<std::string>(test_columns.begin() + label_column_offset,
                                     test_columns.end() - num_ground_truth_columns);


        std::vector<std::shared_ptr<QueryPlan>> queries;
        queries.reserve(num_test_rows);
        for (int i = 0; i < num_test_rows; i++) {
            // DAG
            auto data_source_node  = table_source(db,
                                                 train_table,
                                                 train_table_schema,
                                                 train_table_schema->column_names(),
                                                 maximus::DeviceType::CPU,
                                                 true);
            auto query_source_node = table_source(db,
                                                  test_table,
                                                  test_table_schema,
                                                  test_table_schema->column_names(),
                                                  maximus::DeviceType::CPU,
                                                  true);

            // Project
            std::vector<std::string> query_columns_new = std::vector<std::string>(
                test_columns.begin(), test_columns.end() - num_ground_truth_columns);
            query_columns_new[0] = "query_id";
            auto rename_node =
                rename(query_source_node,
                       std::vector<std::string>(test_columns.begin(),
                                                test_columns.end() - num_ground_truth_columns),
                       query_columns_new);
            // Limit
            auto limit_node =
                std::make_shared<QueryNode>(maximus::EngineType::NATIVE,
                                            maximus::DeviceType::CPU,
                                            maximus::NodeType::LIMIT,
                                            std::make_shared<maximus::LimitProperties>(1, i),
                                            ctx);
            limit_node->add_input(rename_node);

            // Filter
            std::vector<int> test_label_values;
            for (auto& column_name : test_label_columns) {
                auto column = test_table_data.as_table()->get_table()->GetColumnByName(column_name);
                auto label_scalar = column->GetScalar(i);
                CHECK_STATUS(label_scalar.status());
                auto label_scalar_int =
                    std::dynamic_pointer_cast<arrow::Int32Scalar>(label_scalar.ValueOrDie());
                int value = label_scalar_int->value;
                test_label_values.push_back(value);
            }
            auto filter_node = filter(data_source_node,
                                      get_filter_expression_semibound(test_label_values),
                                      maximus::DeviceType::CPU);

            // Vector Search
            auto join_properties = std::make_shared<maximus::VectorJoinExhaustiveProperties>(
                arrow::FieldRef(train_vector_column),
                arrow::FieldRef(test_vector_column),
                K,
                std::nullopt,
                maximus::VectorDistanceMetric::L2,
                false,
                false,
                "distance");
            auto join_node = std::make_shared<QueryNode>(maximus::EngineType::FAISS,
                                                         maximus::DeviceType::CPU,
                                                         maximus::NodeType::VECTOR_JOIN,
                                                         std::move(join_properties),
                                                         ctx);
            join_node->add_input(filter_node);
            join_node->add_input(limit_node);
            queries.push_back(query_plan(table_sink(join_node)));
        }
        return queries;
    }


    std::vector<std::shared_ptr<maximus::QueryPlan>> dispatch(
        const QueryParameters& query_parameters,
        std::shared_ptr<maximus::Database>& db,
        const std::string& test_table,
        const std::string& train_table,
        SchemaPtr test_table_schema,
        SchemaPtr train_table_schema,
        const int num_test_rows) const {
        if (query_parameters.method == "pre") {
            return preFilteringWithFilter(
                db, test_table, train_table, test_table_schema, train_table_schema, num_test_rows);
        } else if (query_parameters.method == "post") {
            return postFiltering(db,
                                 test_table,
                                 train_table,
                                 test_table_schema,
                                 train_table_schema,
                                 num_test_rows,
                                 query_parameters);
        } else if (query_parameters.method == "expr") {
            return postFilteringWithExpression(db,
                                               test_table,
                                               train_table,
                                               test_table_schema,
                                               train_table_schema,
                                               num_test_rows,
                                               query_parameters);
        } else if (query_parameters.method == "bitmap") {
            return preFilteringWithProjectAndIndex(db,
                                                   test_table,
                                                   train_table,
                                                   test_table_schema,
                                                   train_table_schema,
                                                   num_test_rows,
                                                   query_parameters);
        } else if (query_parameters.method == "preFilteringWithProject") {
            return preFilteringWithProject(db,
                                           test_table,
                                           train_table,
                                           test_table_schema,
                                           train_table_schema,
                                           num_test_rows,
                                           query_parameters);
        } else {
            throw std::runtime_error("Unknown implementation method: " + query_parameters.method);
        }
    }
};

/************************************************************************************/
//                                  Unfiltered Workload
/************************************************************************************/

class UnfilteredWorkload : public Workload {
public:
    UnfilteredWorkload(std::shared_ptr<Database>& db): Workload(db){};
    std::vector<std::shared_ptr<maximus::QueryPlan>> withExhaustiveJoin(
        std::shared_ptr<Database>& db,
        const std::string& test_table,
        const std::string& train_table,
        SchemaPtr test_table_schema,
        SchemaPtr train_table_schema,
        const int num_test_rows) const {
        const std::string train_vector_column = "train";  // Holds for all filter-ann datasets
        const std::string test_vector_column  = "test";   // Holds for all filter-ann datasets
        const int K                           = 100;      // Holds for all filter-ann datasets

        auto ctx             = db->get_context();
        auto test_table_data = db->get_table_nocopy(test_table);
        auto test_columns    = test_table_schema->column_names();


        std::vector<std::shared_ptr<QueryPlan>> queries;
        queries.reserve(num_test_rows);
        for (int i = 0; i < num_test_rows; i++) {
            // DAG
            auto data_source_node  = table_source(db,
                                                 train_table,
                                                 train_table_schema,
                                                 train_table_schema->column_names(),
                                                 maximus::DeviceType::CPU,
                                                 true);
            auto query_source_node = table_source(db,
                                                  test_table,
                                                  test_table_schema,
                                                  test_table_schema->column_names(),
                                                  maximus::DeviceType::CPU,
                                                  true);

            // Project
            auto rename_node = rename(
                query_source_node, {"id", test_vector_column}, {"query_id", test_vector_column});
            // Limit
            auto limit_node =
                std::make_shared<QueryNode>(maximus::EngineType::NATIVE,
                                            maximus::DeviceType::CPU,
                                            maximus::NodeType::LIMIT,
                                            std::make_shared<maximus::LimitProperties>(1, i),
                                            ctx);
            limit_node->add_input(rename_node);

            // Vector Search
            auto join_properties = std::make_shared<maximus::VectorJoinExhaustiveProperties>(
                arrow::FieldRef(train_vector_column),
                arrow::FieldRef(test_vector_column),
                K,
                std::nullopt,
                maximus::VectorDistanceMetric::L2,
                false,
                false,
                "distance");
            auto join_node = std::make_shared<QueryNode>(maximus::EngineType::FAISS,
                                                         maximus::DeviceType::CPU,
                                                         maximus::NodeType::VECTOR_JOIN,
                                                         std::move(join_properties),
                                                         ctx);
            join_node->add_input(data_source_node);
            join_node->add_input(limit_node);
            queries.push_back(query_plan(table_sink(join_node)));
        }
        return queries;
    }

    std::vector<std::shared_ptr<maximus::QueryPlan>> withIndex(
        std::shared_ptr<Database>& db,
        const std::string& test_table,
        const std::string& train_table,
        SchemaPtr test_table_schema,
        SchemaPtr train_table_schema,
        const int num_test_rows,
        const QueryParameters& query_parameters) const {
        const std::string train_vector_column = "train";  // Holds for all filter-ann datasets
        const std::string test_vector_column  = "test";   // Holds for all filter-ann datasets
        const int K                           = 100;      // Holds for all filter-ann datasets

        auto ctx             = db->get_context();
        auto test_table_data = db->get_table_nocopy(test_table);

        // Build Index
        auto vector_index = make_index(db, train_table, train_vector_column, query_parameters);
        auto parameters   = make_search_parameters(query_parameters);

        std::vector<std::shared_ptr<QueryPlan>> queries;
        queries.reserve(num_test_rows);
        for (int i = 0; i < num_test_rows; i++) {
            // DAG
            auto data_source_node  = table_source(db,
                                                 train_table,
                                                 train_table_schema,
                                                 train_table_schema->column_names(),
                                                 maximus::DeviceType::CPU,
                                                 true);  // nocopy_variant=true
            auto query_source_node = table_source(db,
                                                  test_table,
                                                  test_table_schema,
                                                  test_table_schema->column_names(),
                                                  maximus::DeviceType::CPU,
                                                  true);

            // Project
            auto rename_node = rename(
                query_source_node, {"id", test_vector_column}, {"query_id", test_vector_column});
            // Limit
            auto limit_node =
                std::make_shared<QueryNode>(maximus::EngineType::NATIVE,
                                            maximus::DeviceType::CPU,
                                            maximus::NodeType::LIMIT,
                                            std::make_shared<maximus::LimitProperties>(1, i),
                                            ctx);
            limit_node->add_input(rename_node);

            // Vector Search
            auto join_properties = std::make_shared<maximus::VectorJoinIndexedProperties>(
                arrow::FieldRef(train_vector_column),
                arrow::FieldRef(test_vector_column),
                vector_index,
                K,
                std::nullopt,
                parameters,
                false,
                false,
                "distance");
            auto join_node = std::make_shared<QueryNode>(maximus::EngineType::FAISS,
                                                         maximus::DeviceType::CPU,
                                                         maximus::NodeType::VECTOR_JOIN,
                                                         std::move(join_properties),
                                                         ctx);
            join_node->add_input(data_source_node);
            join_node->add_input(limit_node);

            queries.push_back(query_plan(table_sink(join_node)));
        }
        return queries;
    }

    std::vector<std::shared_ptr<maximus::QueryPlan>> withProjectVectorDistance(
        std::shared_ptr<Database>& db,
        const std::string& test_table,
        const std::string& train_table,
        SchemaPtr test_table_schema,
        SchemaPtr train_table_schema,
        const int num_test_rows) const {
        const std::string train_vector_column = "train";  // Holds for all filter-ann datasets
        const std::string test_vector_column  = "test";   // Holds for all filter-ann datasets
        const int K                           = 100;      // Holds for all filter-ann datasets

        auto ctx             = db->get_context();
        auto test_table_data = db->get_table_nocopy(test_table);
        auto test_columns    = test_table_schema->column_names();


        std::vector<std::shared_ptr<QueryPlan>> queries;
        queries.reserve(num_test_rows);
        for (int i = 0; i < num_test_rows; i++) {
            // DAG
            auto data_source_node  = table_source(db,
                                                 train_table,
                                                 train_table_schema,
                                                 train_table_schema->column_names(),
                                                 maximus::DeviceType::CPU,
                                                 true);
            auto query_source_node = table_source(db,
                                                  test_table,
                                                  test_table_schema,
                                                  test_table_schema->column_names(),
                                                  maximus::DeviceType::CPU,
                                                  true);
            // Project
            auto rename_node = rename(
                query_source_node, {"id", test_vector_column}, {"query_id", test_vector_column});
            // Limit
            auto limit_node =
                std::make_shared<QueryNode>(maximus::EngineType::NATIVE,
                                            maximus::DeviceType::CPU,
                                            maximus::NodeType::LIMIT,
                                            std::make_shared<maximus::LimitProperties>(1, i),
                                            ctx);
            limit_node->add_input(rename_node);

            // Vector Search
            auto properties = std::make_shared<maximus::VectorProjectDistanceProperties>(
                "distance",
                arrow::FieldRef(train_vector_column),
                arrow::FieldRef(test_vector_column),
                false,
                false);
            auto project_distance_node =
                std::make_shared<QueryNode>(maximus::EngineType::FAISS,
                                            maximus::DeviceType::CPU,
                                            maximus::NodeType::VECTOR_PROJECT_DISTANCE,
                                            std::move(properties),
                                            ctx);
            project_distance_node->add_input(data_source_node);
            project_distance_node->add_input(limit_node);

            auto order_by_node =
                order_by(project_distance_node,
                         {maximus::SortKey("distance", maximus::SortOrder::ASCENDING)},
                         maximus::DeviceType::CPU);

            auto limit_node_final = limit(order_by_node, K, 0);

            queries.push_back(query_plan(table_sink(limit_node_final)));
        }
        return queries;
    }


    std::vector<std::shared_ptr<maximus::QueryPlan>> dispatch(
        const QueryParameters& query_parameters,
        std::shared_ptr<maximus::Database>& db,
        const std::string& test_table,
        const std::string& train_table,
        SchemaPtr test_table_schema,
        SchemaPtr train_table_schema,
        const int num_test_rows) {
        if (query_parameters.method == "index") {
            return withIndex(db,
                             test_table,
                             train_table,
                             test_table_schema,
                             train_table_schema,
                             num_test_rows,
                             query_parameters);
        } else if (query_parameters.method == "exhaustive") {
            return withExhaustiveJoin(
                db, test_table, train_table, test_table_schema, train_table_schema, num_test_rows);
        } else if (query_parameters.method == "projectVectorDistance") {
            return withProjectVectorDistance(
                db, test_table, train_table, test_table_schema, train_table_schema, num_test_rows);
        } else {
            throw std::runtime_error("Unknown implementation method: " + query_parameters.method);
        }
    }
};

/************************************************************************************/
//                                   Benchmarks
/************************************************************************************/

class AgNewsWorkload : public FilterWorkload {
public:
    AgNewsWorkload(std::shared_ptr<Database>& db): FilterWorkload(db){};
    std::vector<std::shared_ptr<QueryPlan>> query_plans(
        const QueryParameters& query_parameters) override {
        return dispatch(query_parameters,
                        db,
                        test_table_name,
                        train_table_name,
                        test_table_schema,
                        train_table_schema,
                        query_points);
    };
    std::vector<std::string> table_names() const override {
        return {test_table_name, train_table_name};
    }
    std::vector<std::shared_ptr<Schema>> table_schemas() const override {
        return {test_table_schema, train_table_schema};
    }
    QualityMetrics evaluate(std::vector<maximus::TablePtr> results) override {
        return evaluate_retrieval_quality(results, db, test_table_name);
    }
    ExpressionPtr get_filter_expression() const override {
        return maximus::expr(maximus::arrow_expr(
            maximus::cp::field_ref("train_label"), "==", maximus::cp::field_ref("test_label")));
    }
    ExpressionPtr get_filter_expression_semibound(std::vector<int> test_labels) const override {
        return maximus::expr(maximus::arrow_expr(
            maximus::cp::field_ref("train_label"), "==", maximus::int32_literal(test_labels[0])));
    }

public:
    inline static const std::string train_table_name = "ag_news_train";
    inline static const std::string test_table_name  = "ag_news_test";
    inline static const SchemaPtr train_table_schema = std::make_shared<Schema>(std::vector(
        {arrow::field("id", arrow::int32(), false),
         arrow::field("train_vec", maximus::embeddings_list(arrow::float32(), 384), false),
         //  arrow::field("train_vec", arrow::fixed_size_list(arrow::float32(), 384), false),
         arrow::field("train_label", arrow::int32(), false)}));
    inline static const SchemaPtr test_table_schema  = std::make_shared<Schema>(std::vector(
        {arrow::field("id", arrow::int32(), false),
          arrow::field("test_vec", maximus::embeddings_list(arrow::float32(), 384), false),
          // arrow::field("test_vec", arrow::fixed_size_list(arrow::float32(), 384), false),
          arrow::field("test_label", arrow::int32(), false),
          arrow::field("neighbors", maximus::embeddings_list(arrow::int32(), 100), false),
          arrow::field("distances", maximus::embeddings_list(arrow::float32(), 100), false)}));
    static const int train_points                    = 120000;
    static const int query_points                    = 7600;
};


class CcNewsWorkload : public FilterWorkload {
public:
    CcNewsWorkload(std::shared_ptr<Database>& db): FilterWorkload(db){};
    std::vector<std::shared_ptr<QueryPlan>> query_plans(
        const QueryParameters& query_parameters) override {
        return dispatch(query_parameters,
                        db,
                        test_table_name,
                        train_table_name,
                        test_table_schema,
                        train_table_schema,
                        query_points);
    };
    std::vector<std::string> table_names() const override {
        return {test_table_name, train_table_name};
    }
    std::vector<std::shared_ptr<Schema>> table_schemas() const override {
        return {test_table_schema, train_table_schema};
    }
    QualityMetrics evaluate(std::vector<maximus::TablePtr> results) override {
        return evaluate_retrieval_quality(results, db, test_table_name);
    }
    ExpressionPtr get_filter_expression() const override {
        return maximus::expr(
            maximus::arrow_between(maximus::cp::field_ref("test_label"),
                                   maximus::arrow_expr(maximus::cp::field_ref("train_label"),
                                                       "-",
                                                       maximus::int32_literal(3 * 24 * 60 * 60)),
                                   maximus::cp::field_ref("train_label")));
    }
    ExpressionPtr get_filter_expression_semibound(std::vector<int> test_labels) const override {
        auto wrapped_test_label = maximus::int32_literal(test_labels[0]);
        return maximus::expr(maximus::arrow_between(
            maximus::cp::field_ref("train_label"),
            maximus::arrow_expr(wrapped_test_label, "-", maximus::int32_literal(3 * 24 * 60 * 60)),
            wrapped_test_label));
    }

public:
    inline static const std::string train_table_name = "cc_news_train";
    inline static const std::string test_table_name  = "cc_news_test";
    inline static const SchemaPtr train_table_schema = std::make_shared<Schema>(std::vector(
        {arrow::field("id", arrow::int32(), false),
         arrow::field("train_vec", maximus::embeddings_list(arrow::float32(), 384), false),
         arrow::field("train_label", arrow::int32(), false)}));
    inline static const SchemaPtr test_table_schema  = std::make_shared<Schema>(std::vector(
        {arrow::field("id", arrow::int32(), false),
          arrow::field("test_vec", maximus::embeddings_list(arrow::float32(), 384), false),
          arrow::field("test_label", arrow::int32(), false),
          arrow::field("neighbors", maximus::embeddings_list(arrow::int32(), 100), false),
          arrow::field("distances", maximus::embeddings_list(arrow::float32(), 100), false)}));
    static const int train_points                    = 630643;
    static const int query_points                    = 10000;
};


class AppReviewsWorkload : public FilterWorkload {
public:
    AppReviewsWorkload(std::shared_ptr<Database>& db): FilterWorkload(db){};
    std::vector<std::shared_ptr<QueryPlan>> query_plans(
        const QueryParameters& query_parameters) override {
        return dispatch(query_parameters,
                        db,
                        test_table_name,
                        train_table_name,
                        test_table_schema,
                        train_table_schema,
                        query_points);
    };
    std::vector<std::string> table_names() const override {
        return {test_table_name, train_table_name};
    }
    std::vector<std::shared_ptr<Schema>> table_schemas() const override {
        return {test_table_schema, train_table_schema};
    }
    QualityMetrics evaluate(std::vector<maximus::TablePtr> results) override {
        return evaluate_retrieval_quality(results, db, test_table_name);
    }
    ExpressionPtr get_filter_expression() const override {
        return get_filter_expression_internal({maximus::cp::field_ref("test_label_0"),
                                               maximus::cp::field_ref("test_label_1"),
                                               maximus::cp::field_ref("test_label_2")});
    }
    ExpressionPtr get_filter_expression_semibound(std::vector<int> test_labels) const override {
        assert(test_labels.size() == 3);
        return get_filter_expression_internal({maximus::int32_literal(test_labels[0]),
                                               maximus::int32_literal(test_labels[1]),
                                               maximus::int32_literal(test_labels[2])});
    }
    ExpressionPtr get_filter_expression_internal(std::vector<arrow::Expression> test_labels) const {
        assert(test_labels.size() == 3);
        auto test_text_length  = test_labels[0];
        auto train_text_length = maximus::cp::field_ref("train_label_0");
        auto test_unixtime     = test_labels[1];
        auto train_unixtime    = maximus::cp::field_ref("train_label_1");
        auto test_star         = test_labels[2];
        auto train_star        = maximus::cp::field_ref("train_label_2");

        auto time_shifted =
            maximus::arrow_expr(test_unixtime, "-", maximus::int32_literal(30 * 24 * 60 * 60));
        auto text_plus  = maximus::arrow_expr(test_text_length, "+", maximus::int32_literal(30));
        auto text_minus = maximus::arrow_expr(test_text_length, "-", maximus::int32_literal(30));
        auto star_lower = maximus::cp::call("choose",
                                            {test_star,
                                             maximus::int32_literal(0),
                                             maximus::int32_literal(1),
                                             maximus::int32_literal(1),
                                             maximus::int32_literal(3),
                                             maximus::int32_literal(3),
                                             maximus::int32_literal(4)});
        auto star_upper = maximus::cp::call("choose",
                                            {test_star,
                                             maximus::int32_literal(0),
                                             maximus::int32_literal(2),
                                             maximus::int32_literal(2),
                                             maximus::int32_literal(4),
                                             maximus::int32_literal(5),
                                             maximus::int32_literal(5)});

        return maximus::expr(
            maximus::arrow_all({maximus::arrow_between(train_text_length, text_minus, text_plus),
                                maximus::arrow_between(train_unixtime, time_shifted, test_unixtime),
                                maximus::arrow_between(train_star, star_lower, star_upper)}));
    }

public:
    inline static const std::string train_table_name = "app_reviews_train";
    inline static const std::string test_table_name  = "app_reviews_test";
    inline static const SchemaPtr train_table_schema = std::make_shared<Schema>(std::vector(
        {arrow::field("id", arrow::int32(), false),
         arrow::field("train_vec", maximus::embeddings_list(arrow::float32(), 384), false),
         arrow::field("train_label_0", arrow::int32(), false),
         arrow::field("train_label_1", arrow::int32(), false),
         arrow::field("train_label_2", arrow::int32(), false)}));
    inline static const SchemaPtr test_table_schema  = std::make_shared<Schema>(std::vector(
        {arrow::field("id", arrow::int32(), false),
          arrow::field("test_vec", maximus::embeddings_list(arrow::float32(), 384), false),
          arrow::field("test_label_0", arrow::int32(), false),
          arrow::field("test_label_1", arrow::int32(), false),
          arrow::field("test_label_2", arrow::int32(), false),
          arrow::field("neighbors", maximus::embeddings_list(arrow::int32(), 100), false),
          arrow::field("distances", maximus::embeddings_list(arrow::float32(), 100), false)}));
    static const int train_points                    = 277936;
    static const int query_points                    = 10000;
};


class AmazonWorkload : public FilterWorkload {
public:
    AmazonWorkload(std::shared_ptr<Database>& db): FilterWorkload(db){};
    std::vector<std::shared_ptr<QueryPlan>> query_plans(
        const QueryParameters& query_parameters) override {
        return dispatch(query_parameters,
                        db,
                        test_table_name,
                        train_table_name,
                        test_table_schema,
                        train_table_schema,
                        query_points);
    };
    std::vector<std::string> table_names() const override {
        return {test_table_name, train_table_name};
    }
    std::vector<std::shared_ptr<Schema>> table_schemas() const override {
        return {test_table_schema, train_table_schema};
    }
    QualityMetrics evaluate(std::vector<maximus::TablePtr> results) override {
        return evaluate_retrieval_quality(results, db, test_table_name);
    }
    ExpressionPtr get_filter_expression() const override {
        return get_filter_expression_internal({maximus::cp::field_ref("test_label_0"),
                                               maximus::cp::field_ref("test_label_1"),
                                               maximus::cp::field_ref("test_label_2"),
                                               maximus::cp::field_ref("test_label_3"),
                                               maximus::cp::field_ref("test_label_4")});
    }
    ExpressionPtr get_filter_expression_semibound(std::vector<int> test_labels) const override {
        return get_filter_expression_internal({maximus::int32_literal(test_labels[0]),
                                               maximus::int32_literal(test_labels[1]),
                                               maximus::int32_literal(test_labels[2]),
                                               maximus::int32_literal(test_labels[3]),
                                               maximus::int32_literal(test_labels[4])});
    }
    ExpressionPtr get_filter_expression_internal(std::vector<arrow::Expression> test_labels) const {
        // Field references
        auto test_unixtime  = test_labels[0];
        auto train_unixtime = maximus::cp::field_ref("train_label_0");
        auto test_str_len   = test_labels[1];
        auto train_str_len  = maximus::cp::field_ref("train_label_1");
        auto test_asin      = test_labels[2];
        auto train_asin     = maximus::cp::field_ref("train_label_2");
        auto test_overall   = test_labels[3];
        auto train_overall  = maximus::cp::field_ref("train_label_3");
        auto test_vote      = test_labels[4];
        auto train_vote     = maximus::cp::field_ref("train_label_4");

        // Constants
        const int asin_min = 1048767;
        const int asin_max = 2147483647;
        const int asin_scp = asin_max - asin_min;

        // Time range: last 7 days
        auto time_shifted =
            maximus::arrow_expr(train_unixtime, "-", maximus::int32_literal(7 * 24 * 60 * 60));

        // str_len +/- 30
        auto str_len_minus = maximus::arrow_expr(train_str_len, "-", maximus::int32_literal(30));
        auto str_len_plus  = maximus::arrow_expr(train_str_len, "+", maximus::int32_literal(30));

        // asin ± 5% range
        auto asin_delta = maximus::int32_literal(static_cast<int>(0.05 * asin_scp));
        auto asin_minus = maximus::arrow_expr(train_asin, "-", asin_delta);
        auto asin_plus  = maximus::arrow_expr(train_asin, "+", asin_delta);

        // overall rating bucketing
        auto overall_lower = maximus::cp::call("choose",
                                               {test_overall,
                                                maximus::int32_literal(0),
                                                maximus::int32_literal(1),
                                                maximus::int32_literal(1),
                                                maximus::int32_literal(3),
                                                maximus::int32_literal(3),
                                                maximus::int32_literal(4)});

        auto overall_upper = maximus::cp::call("choose",
                                               {test_overall,
                                                maximus::int32_literal(0),
                                                maximus::int32_literal(2),
                                                maximus::int32_literal(2),
                                                maximus::int32_literal(4),
                                                maximus::int32_literal(5),
                                                maximus::int32_literal(5)});

        // vote threshold bucket
        auto vote_threshold = maximus::arrow_expr(train_vote, "/", maximus::int32_literal(10));

        // Compose all filters using arrow_between and comparisons
        return maximus::expr(
            maximus::arrow_all({maximus::arrow_between(test_unixtime, time_shifted, train_unixtime),
                                maximus::arrow_between(test_str_len, str_len_minus, str_len_plus),
                                maximus::arrow_between(test_asin, asin_minus, asin_plus),
                                maximus::arrow_between(test_overall, overall_lower, overall_upper),
                                maximus::arrow_expr(test_vote, ">=", vote_threshold)}));
    }

public:
    // TODO: Tese the amazon dataset, which would have ot be LargeListArray (for int64 offsets)
    inline static const std::string train_table_name =
        "amazon_train_ll";  // _ll indicates that it's the LargeListArray version
    inline static const std::string test_table_name  = "amazon_test_ll";
    inline static const SchemaPtr train_table_schema = std::make_shared<Schema>(std::vector(
        {arrow::field("id", arrow::int64(), false),
         //  arrow::field("train_vec", arrow::fixed_size_list(arrow::float32(), 384), false), // FixedSizeListArray
         arrow::field("train_vec", arrow::large_list(arrow::float32()), false),  // LargeListArray
         arrow::field("train_label_0", arrow::int32(), false),
         arrow::field("train_label_1", arrow::int32(), false),
         arrow::field("train_label_2", arrow::int32(), false),
         arrow::field("train_label_3", arrow::int32(), false),
         arrow::field("train_label_4", arrow::int32(), false)}));
    inline static const SchemaPtr test_table_schema  = std::make_shared<Schema>(std::vector(
        {arrow::field("id", arrow::int64(), false),
          // arrow::field("test_vec", arrow::fixed_size_list(arrow::float32(), 384), false),
          arrow::field("test_vec", arrow::large_list(arrow::float32()), false),
          arrow::field("test_label_0", arrow::int32(), false),
          arrow::field("test_label_1", arrow::int32(), false),
          arrow::field("test_label_2", arrow::int32(), false),
          arrow::field("test_label_3", arrow::int32(), false),
          arrow::field("test_label_4", arrow::int32(), false),
          //   arrow::field("neighbors", arrow::fixed_size_list(arrow::int32(), 100), false),
          //   arrow::field("distances", arrow::fixed_size_list(arrow::float32(), 100), false)
          arrow::field("neighbors", arrow::large_list(arrow::int32()), false),
          arrow::field("distances", arrow::large_list(arrow::float32()), false)}));
    static const int train_points                    = 15928208;
    static const int query_points                    = 10000;
};


class AgNewsUnfilteredWorkload : public UnfilteredWorkload {
public:
    AgNewsUnfilteredWorkload(std::shared_ptr<Database>& db): UnfilteredWorkload(db){};
    std::vector<std::shared_ptr<QueryPlan>> query_plans(
        const QueryParameters& query_parameters) override {
        return dispatch(query_parameters,
                        db,
                        test_table_name,
                        train_table_name,
                        test_table_schema,
                        train_table_schema,
                        query_points);
    };
    std::vector<std::string> table_names() const override {
        return {test_table_name, train_table_name};
    }
    std::vector<std::shared_ptr<Schema>> table_schemas() const override {
        return {test_table_schema, train_table_schema};
    }
    QualityMetrics evaluate(std::vector<maximus::TablePtr> results) override {
        return evaluate_retrieval_quality(results, db, test_table_name);
    }

public:
    inline static const std::string train_table_name = "ag_news_unfiltered_train";
    inline static const std::string test_table_name  = "ag_news_unfiltered_test";
    inline static const SchemaPtr train_table_schema = std::make_shared<Schema>(std::vector(
        {arrow::field("id", arrow::int32(), false),
         //  arrow::field("train", maximus::embeddings_list(arrow::float32(), 384), false)}));
         arrow::field("train", arrow::fixed_size_list(arrow::float32(), 384), false)}));
    inline static const SchemaPtr test_table_schema  = std::make_shared<Schema>(std::vector(
        {arrow::field("id", arrow::int32(), false),
          //  arrow::field("test", maximus::embeddings_list(arrow::float32(), 384), false),
          arrow::field("test", arrow::fixed_size_list(arrow::float32(), 384), false),
          arrow::field("neighbors", maximus::embeddings_list(arrow::int32(), 100), false),
          arrow::field("distances", maximus::embeddings_list(arrow::float32(), 100), false)}));
    static const int train_points                    = 120000;
    static const int query_points                    = 7600;
};


/************************************************************************************/
//                                   OTHER
/************************************************************************************/


std::shared_ptr<AbstractWorkload> get_workload(const std::string& benchmark,
                                               std::shared_ptr<Database>& db) {
    if (benchmark == "ag_news-384-euclidean-filter")
        return std::make_shared<AgNewsWorkload>(db);
    else if (benchmark == "cc_news-384-euclidean-filter")
        return std::make_shared<CcNewsWorkload>(db);
    else if (benchmark == "app_reviews-384-euclidean-filter")
        return std::make_shared<AppReviewsWorkload>(db);
    else if (benchmark == "amazon-384-euclidean-5filter")
        return std::make_shared<AmazonWorkload>(db);
    else if (benchmark == "ag_news-384-euclidean")
        return std::make_shared<AgNewsUnfilteredWorkload>(db);
    else
        throw std::runtime_error("Unknown benchmark: " + benchmark);
}


void write_results_to_file(maximus::Context& context, std::vector<maximus::TablePtr> results) {
    std::string target_name = "results_out.csv";
    std::ofstream file;
    file.open(target_name);
    for (auto& table : results) {
        file << table->to_string();
    }
    file.close();
    std::cout << "Query results saved to " << target_name << std::endl;
}

}  // namespace big_vector_bench
