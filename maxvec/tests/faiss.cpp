#include <faiss/AutoTune.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/index_factory.h>
#include <faiss/index_io.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <maximus/database.hpp>
#include <maximus/database_catalogue.hpp>
#include <maximus/frontend/expressions.hpp>
#include <maximus/frontend/query_plan_api.hpp>
#include <maximus/indexes/index.hpp>
#include <maximus/operators/faiss/interop.hpp>
#include <maximus/operators/faiss/join_exhaustive_operator.hpp>
#include <maximus/operators/faiss/join_indexed_operator.hpp>
#include <maximus/operators/faiss/project_distance_operator.hpp>
#include <maximus/types/expression.hpp>
#include <maximus/types/types.hpp>

// fixtures
#include "sift_fixture.hpp"

using idx_t        = faiss::idx_t;
using SchemaPtr    = std::shared_ptr<maximus::Schema>;
using SchemaVector = std::vector<SchemaPtr>;

namespace test {

struct FaissTestContext {
    FaissTestContext() {
        CHECK_STATUS(maximus::TableBatch::from_json(context,
                                                    data_schema,
                                                    {R"([
            [[1.00, 1.00, 1.00, 1.00, 1.00], "a"],
            [[1.00, 3.00, 1.00, 3.00, 1.00], "b"],
            [[2.00, 2.00, 8.00, 8.00, 8.00], "c"],
            [[8.00, 8.00, 2.00, 2.00, 2.00], "d"],
            [[3.00, 1.00, 3.00, 1.00, 3.00], "e"]
        ])"},
                                                    data_table));
        CHECK_STATUS(maximus::TableBatch::from_json(context,
                                                    query_schema,
                                                    {R"([
            [0, [0.99, 1.01, 0.98, 1.02, 0.99]],
            [1, [8.05, 7.55, 1.95, 2.10, 1.99]]
        ])"},
                                                    query_table));
    }
    arrow::FieldVector data_fields = {
        arrow::field("dvector", maximus::embeddings_list(arrow::float32(), 5)),  //  a 5d emb
        arrow::field("dcategory", arrow::utf8())                                 //  a string
    };
    arrow::FieldVector query_fields = {
        arrow::field("id", arrow::int32()),
        arrow::field("qvector", maximus::embeddings_list(arrow::float32(), 5))  //  a 5d emb
    };
    SchemaPtr data_schema    = std::make_shared<maximus::Schema>(data_fields);
    SchemaPtr query_schema   = std::make_shared<maximus::Schema>(query_fields);
    maximus::Context context = maximus::make_context();
    maximus::TableBatchPtr data_table;
    maximus::TableBatchPtr query_table;
};
FaissTestContext ftc;

TEST(faiss, FaissRunsProperly) {
    int d           = 64;    // dimension
    faiss::idx_t nb = 1000;  // database size
    faiss::idx_t nq = 10;    // number of queries
    int k           = 100;   // top-k
    int nrun        = 1;

    //std::cout << "========== d=" << d << " nb=" << nb << " nq=" << nq << " ==========" << std::endl;

    // Generate synthetic data (uniform [0, 1])
    std::vector<float> xb(nb * d);
    std::vector<float> xq(nq * d);

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> distrib;

    for (size_t i = 0; i < xb.size(); ++i) xb[i] = distrib(rng);
    for (size_t i = 0; i < xq.size(); ++i) xq[i] = distrib(rng);

    // Create index
    faiss::IndexFlatL2 index(d);
    index.add(nb, xb.data());

    // Prepare output buffers
    std::vector<faiss::idx_t> I(nq * k);
    std::vector<float> D(nq * k);

    // Run search repeatedly
    for (int run = 0; run < nrun; ++run) {
        auto start = std::chrono::high_resolution_clock::now();
        index.search(nq, xq.data(), k, D.data(), I.data());
        auto end       = std::chrono::high_resolution_clock::now();
        float duration = std::chrono::duration<float>(end - start).count();

        //std::cout << "Run " << run << ": " << duration << " s" << std::endl;
    }
}

TEST(faiss, ListTypeFromJson) {
    auto fields = {
        arrow::field("vector", maximus::embeddings_list(arrow::float32(), 20)),  //  a 20d embedding
        arrow::field("label", arrow::utf8())};
    auto input_schema = std::make_shared<maximus::Schema>(fields);

    auto context = maximus::make_context();
    maximus::TablePtr table;
    auto status = maximus::Table::from_json(context,
                                            input_schema,
                                            {R"([
            [[0.14, 0.55, 0.85, 0.23, 0.91, 0.12, 0.34, 0.45, 0.67, 0.89, 0.10, 0.22, 0.33, 0.44, 0.55, 0.66, 0.77, 0.88, 0.99, 0.11], "x"],
            [[0.14, 0.55, 0.85, 0.23, 0.91, 0.12, 0.34, 0.45, 0.67, 0.89, 0.10, 0.22, 0.33, 0.44, 0.55, 0.66, 0.77, 0.88, 0.99, 0.11], "y"]
        ])"},
                                            table);
    CHECK_STATUS(status);
    //table->print();
}

TEST(faiss, ListTypeFromParquet) {
    auto context = maximus::make_context();
    maximus::TablePtr table;
    auto status = maximus::Table::from_parquet(
        context,
        std::string(PROJECT_SOURCE_DIR) + "/tests/sift/parquet10K/learn.parquet",
        nullptr,
        {},
        table);
    CHECK_STATUS(status);
    // table->slice(0, 5)->print();
}

TEST(faiss, JoinOperator) {
    // ===============================================
    //     CREATING THE INPUT
    // ===============================================
    auto fields = {
        arrow::field("dvector", maximus::embeddings_list(arrow::float32(), 5)),  //  a 5d emb
        arrow::field("category", arrow::utf8())                                  //  a string
    };
    auto query_fields = {
        arrow::field("id", arrow::int32()),
        arrow::field("qvector", maximus::embeddings_list(arrow::float32(), 5))  //  a 5d emb
    };
    auto input_schema  = std::make_shared<maximus::Schema>(fields);
    auto query_schema  = std::make_shared<maximus::Schema>(query_fields);
    auto result_schema = std::make_shared<maximus::Schema>(arrow::FieldVector{
        arrow::field("id", arrow::int32()), arrow::field("category", arrow::utf8())});

    auto context = maximus::make_context();
    maximus::TableBatchPtr table;
    maximus::TableBatchPtr query_vector;

    auto status = maximus::TableBatch::from_json(context,
                                                 input_schema,
                                                 {R"([
            [[1.00, 1.00, 1.00, 1.00, 1.00], "a"],
            [[1.00, 3.00, 1.00, 3.00, 1.00], "b"],
            [[2.00, 2.00, 8.00, 8.00, 8.00], "c"],
            [[8.00, 8.00, 2.00, 2.00, 2.00], "d"],
            [[3.00, 1.00, 3.00, 1.00, 3.00], "e"]
        ])"},
                                                 table);
    CHECK_STATUS(status);
    status = maximus::TableBatch::from_json(context,
                                            query_schema,
                                            {R"([
            [0, [0.99, 1.01, 0.98, 1.02, 0.99]],
            [1, [8.05, 7.55, 1.95, 2.10, 1.99]]
        ])"},
                                            query_vector);
    CHECK_STATUS(status);

    // ===============================================
    //     CREATING THE OPERATOR
    // ===============================================
    auto properties = std::make_shared<maximus::VectorJoinExhaustiveProperties>(
        arrow::FieldRef("dvector"), arrow::FieldRef("qvector"), 2);  // k = 2
    std::vector<std::shared_ptr<maximus::Schema>> input_schemas = {std::move(input_schema),
                                                                   std::move(query_schema)};

    auto join_operator = std::make_shared<maximus::faiss::JoinExhaustiveOperator>(
        context, std::move(input_schemas), properties);
    join_operator->next_op_type     = maximus::PhysicalOperatorType::TABLE_SINK;
    join_operator->next_engine_type = maximus::EngineType::NATIVE;
    std::cout << "operator = \n" << join_operator->to_string() << std::endl;

    // ===============================================
    //     PUSH THE BATCH TO THE OPERATOR
    // ===============================================
    join_operator->add_input(maximus::DeviceTablePtr(std::move(table)), 0);
    join_operator->no_more_input(0);
    join_operator->add_input(maximus::DeviceTablePtr(std::move(query_vector)), 1);
    join_operator->no_more_input(1);
    maximus::DeviceTablePtr output = join_operator->export_next_batch();
    output.convert_to<maximus::TablePtr>(context, result_schema);
    std::string result = output.as_table()->to_string();
    EXPECT_STREQ(result.data(),
                 "id(int32), category(utf8)\n"
                 "0, a\n"
                 "0, b\n"
                 "1, d\n"
                 "1, e\n");
}

TEST(faiss, JoinOperatorWithCallbackPostfiltering) {
    // Parameters
    std::string index_string = "Flat";
    std::string path         = std::string(PROJECT_SOURCE_DIR);
    auto db_catalogue        = maximus::make_catalogue(path);
    auto db                  = maximus::make_database(db_catalogue);

    // ===============================================
    //     CREATING THE INPUT
    // ===============================================
    auto data_schema                 = std::make_shared<maximus::Schema>(arrow::FieldVector{
        arrow::field("dvector", maximus::embeddings_list(arrow::float32(), 5)),  //  a 5d emb
        arrow::field("category", arrow::utf8())});
    auto data_schema_without_vectors = std::make_shared<maximus::Schema>(
        arrow::FieldVector{arrow::field("category", arrow::utf8())});
    auto query_schema = std::make_shared<maximus::Schema>(
        arrow::FieldVector{arrow::field("qid", arrow::int32()),
                           arrow::field("qvector", maximus::embeddings_list(arrow::float32(), 5))});
    auto result_schema = std::make_shared<maximus::Schema>(
        arrow::FieldVector{arrow::field("qid", arrow::int32()),
                           arrow::field("dvector", maximus::embeddings_list(arrow::float32(), 5)),
                           arrow::field("category", arrow::utf8())});

    auto context = maximus::make_context();
    maximus::TableBatchPtr table;
    maximus::TableBatchPtr table_without_vectors;
    maximus::TableBatchPtr query_vector;

    auto status = maximus::TableBatch::from_json(context,
                                                 data_schema,
                                                 {R"([
            [[1.00, 1.00, 1.00, 1.00, 1.00], "a"],
            [[1.00, 3.00, 1.00, 3.00, 1.00], "b"],
            [[8.00, 8.00, 2.00, 2.00, 2.00], "a"],
            [[2.00, 2.00, 8.00, 8.00, 8.00], "b"],
            [[3.00, 1.00, 3.00, 1.00, 3.00], "a"]
        ])"},
                                                 table);
    CHECK_STATUS(status);
    status = maximus::TableBatch::from_json(context,
                                            data_schema_without_vectors,
                                            {R"([
            ["a"],
            ["b"],
            ["a"],
            ["b"],
            ["a"]
        ])"},
                                            table_without_vectors);
    CHECK_STATUS(status);
    status = maximus::TableBatch::from_json(context,
                                            query_schema,
                                            {R"([
            [0, [0.99, 1.01, 0.98, 1.02, 0.99]],
            [1, [8.05, 7.55, 1.95, 2.10, 1.99]]
        ])"},
                                            query_vector);
    CHECK_STATUS(status);


    // ===============================================
    //     SETTING UP THE INDEX
    // ===============================================
    const int d       = 5;
    auto vector_index = maximus::faiss::FaissIndex::factory_make(
        db->get_context(), d, index_string, maximus::faiss::MetricType::METRIC_L2);
    auto column  = table->get_table_batch()->GetColumnByName("dvector");
    auto vectors = std::static_pointer_cast<maximus::EmbeddingsArray>(column);
    vector_index->train(*vectors);
    vector_index->add(*vectors);
    auto inner_params = std::make_shared<::faiss::SearchParametersIVF>();
    auto params = std::make_shared<maximus::faiss::FaissSearchParameters>(std::move(inner_params));

    // ===============================================
    //     CREATING THE OPERATOR
    // ===============================================
    std::shared_ptr<maximus::Expression> predicate = maximus::expr(maximus::arrow_expr(
        arrow::compute::field_ref("category"), "==", maximus::string_literal("a")));
    auto properties =
        std::make_shared<maximus::VectorJoinIndexedProperties>(arrow::FieldRef("dvector"),
                                                               arrow::FieldRef("qvector"),
                                                               vector_index,
                                                               2,  // k = 2
                                                               std::nullopt,
                                                               params,
                                                               true,   // keep dvector
                                                               false,  // keep qvector
                                                               std::nullopt,
                                                               predicate);
    //std::vector<std::shared_ptr<maximus::Schema>> input_schemas = {
    //    std::move(input_schema_without_vectors), std::move(query_schema)};
    std::vector<std::shared_ptr<maximus::Schema>> input_schemas = {std::move(data_schema),
                                                                   std::move(query_schema)};

    auto join_operator = std::make_shared<maximus::faiss::JoinIndexedOperator>(
        db->get_context(), std::move(input_schemas), properties);
    join_operator->next_op_type     = maximus::PhysicalOperatorType::TABLE_SINK;
    join_operator->next_engine_type = maximus::EngineType::NATIVE;

    // ===============================================
    //     PUSH THE BATCH TO THE OPERATOR
    // ===============================================
    // join_operator->add_input(maximus::DeviceTablePtr(std::move(table_without_vectors)), 0);
    join_operator->add_input(maximus::DeviceTablePtr(std::move(table)), 0);
    join_operator->add_input(maximus::DeviceTablePtr(std::move(query_vector)), 1);
    join_operator->no_more_input(0);
    join_operator->no_more_input(1);
    std::cout << "operator = \n" << join_operator->to_string() << std::endl;
    maximus::DeviceTablePtr output = join_operator->export_next_batch();
    output.convert_to<maximus::TablePtr>(context, result_schema);
    std::string result = output.as_table()->to_string();
    EXPECT_STREQ(result.data(),
                 "qid(int32), dvector(list), category(utf8)\n"
                 "0, [1.000000, 1.000000, ..., 1.000000], a\n"
                 "0, [3.000000, 1.000000, ..., 3.000000], a\n"
                 "1, [8.000000, 8.000000, ..., 2.000000], a\n"
                 "1, [3.000000, 1.000000, ..., 3.000000], a\n");
}

TEST(faiss, JoinOperatorWithBitmapPrefiltering) {
    // ===============================================
    //     CREATING THE INPUT
    // ===============================================
    auto input_schema = std::make_shared<maximus::Schema>(arrow::FieldVector{
        arrow::field("dvector", maximus::embeddings_list(arrow::float32(), 5)),  //  a 5d emb
        arrow::field("category", arrow::utf8()),
        arrow::field("filter", arrow::boolean())});  //  a bitmap filter
    auto query_schema = std::make_shared<maximus::Schema>(
        arrow::FieldVector{arrow::field("id", arrow::int32()),
                           arrow::field("qvector", maximus::embeddings_list(arrow::float32(), 5))});
    auto result_schema = std::make_shared<maximus::Schema>(arrow::FieldVector{
        arrow::field("id", arrow::int32()), arrow::field("category", arrow::utf8())});

    auto context = maximus::make_context();
    maximus::TableBatchPtr table;
    maximus::TableBatchPtr query_vector;

    auto status = maximus::TableBatch::from_json(context,
                                                 input_schema,
                                                 {R"([
            [[1.00, 1.00, 1.00, 1.00, 1.00], "a", true],
            [[1.00, 3.00, 1.00, 3.00, 1.00], "a", true],
            [[2.00, 2.00, 8.00, 8.00, 8.00], "c", false],
            [[8.00, 8.00, 2.00, 2.00, 2.00], "d", false],
            [[3.00, 1.00, 3.00, 1.00, 3.00], "e", false]
        ])"},
                                                 table);
    CHECK_STATUS(status);
    status = maximus::TableBatch::from_json(context,
                                            query_schema,
                                            {R"([
            [0, [0.99, 1.01, 0.98, 1.02, 0.99]],
            [1, [8.05, 7.55, 1.95, 2.10, 1.99]]
        ])"},
                                            query_vector);
    CHECK_STATUS(status);

    // ===============================================
    //     CREATING THE OPERATOR
    // ===============================================
    std::shared_ptr<maximus::Expression> predicate = maximus::expr(maximus::arrow_expr(
        arrow::compute::field_ref("category"), "==", maximus::string_literal("a")));
    auto properties =
        std::make_shared<maximus::VectorJoinExhaustiveProperties>(arrow::FieldRef("dvector"),
                                                                  arrow::FieldRef("qvector"),
                                                                  2,  // k = 2,
                                                                  std::nullopt,
                                                                  maximus::VectorDistanceMetric::L2,
                                                                  false,
                                                                  false,
                                                                  std::nullopt,
                                                                  arrow::FieldRef("filter"));
    std::vector<std::shared_ptr<maximus::Schema>> input_schemas = {std::move(input_schema),
                                                                   std::move(query_schema)};

    auto join_operator = std::make_shared<maximus::faiss::JoinExhaustiveOperator>(
        context, std::move(input_schemas), properties);
    join_operator->next_op_type     = maximus::PhysicalOperatorType::TABLE_SINK;
    join_operator->next_engine_type = maximus::EngineType::NATIVE;

    // ===============================================
    //     PUSH THE BATCH TO THE OPERATOR
    // ===============================================
    join_operator->add_input(maximus::DeviceTablePtr(std::move(table)), 0);
    join_operator->add_input(maximus::DeviceTablePtr(std::move(query_vector)), 1);
    join_operator->no_more_input(0);

    maximus::DeviceTablePtr output = join_operator->export_next_batch();
    output.convert_to<maximus::TablePtr>(context, result_schema);
    std::string result = output.as_table()->to_string();
    EXPECT_STREQ(result.data(),
                 "id(int32), category(utf8)\n"
                 "0, a\n"
                 "0, a\n"
                 "1, a\n"
                 "1, a\n");
}

TEST(faiss, JoinOperatorWithIndex) {
    // Parameters
    std::string index_string = "IVF48,Flat";  // 48 centroids
    std::string path         = std::string(PROJECT_SOURCE_DIR) + "/tests/sift/parquet10K";

    // ===============================================
    //     CREATING THE INPUT
    // ===============================================
    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);

    auto data_table_schema =
        std::make_shared<maximus::Schema>(std::vector<std::shared_ptr<arrow::Field>>{
            arrow::field("id", arrow::int32()),
            arrow::field("vector", maximus::embeddings_list(arrow::int32(), 128))});
    auto query_table_schema =
        std::make_shared<maximus::Schema>(std::vector<std::shared_ptr<arrow::Field>>{
            arrow::field("id", arrow::int32()),
            arrow::field("vector", maximus::embeddings_list(arrow::int32(), 128))});

    std::vector<std::shared_ptr<maximus::Schema>> input_schemas = {data_table_schema,
                                                                   query_table_schema};

    db->load_table("base", data_table_schema, {"id", "vector"}, maximus::DeviceType::CPU);
    db->load_table("query", query_table_schema, {"id", "vector"}, maximus::DeviceType::CPU);
    auto data_table  = db->get_table("base");
    auto query_table = db->get_table("query");

    data_table.convert_to<maximus::TableBatchPtr>(db->get_context(), input_schemas[0]);
    query_table.convert_to<maximus::TableBatchPtr>(db->get_context(), input_schemas[1]);

    // ===============================================
    //     SETTING UP THE INDEX
    // ===============================================
    auto column  = data_table.as_table_batch()->get_table_batch()->GetColumnByName("vector");
    auto vectors = std::static_pointer_cast<maximus::EmbeddingsArray>(
        column);  // maximus::embeddings_values(column);
    auto field_type = input_schemas[0]->column_types()["vector"];
    std::cout << "vector type = " << field_type->ToString() << std::endl;
    const int d = maximus::embedding_dimension(column);
    EXPECT_EQ(d, 128);
    std::cout << "d = " << d << std::endl;
    auto vector_index = maximus::faiss::FaissIndex::factory_make(
        db->get_context(), d, index_string, maximus::faiss::MetricType::METRIC_L2);

    vector_index->train(*vectors);
    vector_index->add(*vectors);

    //db->add_index("base", "vector", vector_index);
    //auto vector_index = db->get_indexes("base", "vector")[0];

    // ===============================================
    //     CREATING THE OPERATOR
    // ===============================================
    auto inner_params = std::make_shared<::faiss::SearchParametersIVF>();
    auto params = std::make_shared<maximus::faiss::FaissSearchParameters>(std::move(inner_params));
    auto properties =
        std::make_shared<maximus::VectorJoinIndexedProperties>(arrow::FieldRef("vector"),
                                                               arrow::FieldRef("vector"),
                                                               vector_index,
                                                               2,             // k = 2,
                                                               std::nullopt,  // radius
                                                               std::move(params));

    auto join_operator = std::make_shared<maximus::faiss::JoinIndexedOperator>(
        db->get_context(), std::move(input_schemas), properties);
    join_operator->next_op_type     = maximus::PhysicalOperatorType::TABLE_SINK;
    join_operator->next_engine_type = maximus::EngineType::NATIVE;

    // ===============================================
    //     PUSH THE BATCH TO THE OPERATOR
    // ===============================================
    join_operator->add_input(data_table, 0);
    join_operator->add_input(query_table, 1);
    join_operator->no_more_input(1);
    join_operator->no_more_input(0);
    maximus::DeviceTablePtr output = join_operator->export_next_batch();
    //std::string result = maximus::Table(db->get_context(), output.as_arrow_table()).to_string();
}

TEST(faiss, JoinOperatorStreaming) {
    // CREATE OPERATOR
    auto properties = std::make_shared<maximus::VectorJoinExhaustiveProperties>(
        arrow::FieldRef("dvector"), arrow::FieldRef("qvector"), 2);  // k = 2
    SchemaVector input_schemas = {ftc.data_schema, ftc.query_schema};
    auto join_operator         = std::make_shared<maximus::faiss::JoinExhaustiveOperator>(
        ftc.context, std::move(input_schemas), properties);
    join_operator->next_op_type     = maximus::PhysicalOperatorType::TABLE_SINK;
    join_operator->next_engine_type = maximus::EngineType::NATIVE;
    SchemaPtr result_schema         = std::make_shared<maximus::Schema>(arrow::FieldVector{
        arrow::field("id", arrow::int32()), arrow::field("category", arrow::utf8())});
    //std::cout << "operator = \n" << join_operator->to_string() << std::endl;

    // PUSH THE BATCH TO THE OPERATOR
    join_operator->add_input(maximus::DeviceTablePtr(ftc.query_table), 1);
    join_operator->add_input(maximus::DeviceTablePtr(ftc.data_table), 0);
    join_operator->add_input(maximus::DeviceTablePtr(ftc.query_table), 1);
    EXPECT_FALSE(join_operator->has_more_batches(0));
    join_operator->no_more_input(0);
    EXPECT_TRUE(join_operator->has_more_batches(0));
    join_operator->add_input(maximus::DeviceTablePtr(ftc.query_table), 1);
    join_operator->no_more_input(1);
    maximus::TablePtr output = join_operator->export_table();
    EXPECT_EQ(output->num_rows(), 2 * 3 * ftc.query_table->num_rows());  //output->print();
}

TEST(faiss, JoinOperatorWithDistanceColumn) {
    // CREATE OPERATOR
    auto properties =
        std::make_shared<maximus::VectorJoinExhaustiveProperties>(arrow::FieldRef("dvector"),
                                                                  arrow::FieldRef("qvector"),
                                                                  2,  // k = 2,
                                                                  std::nullopt,
                                                                  maximus::VectorDistanceMetric::L2,
                                                                  false,
                                                                  false,
                                                                  "distance");
    SchemaVector input_schemas = {ftc.data_schema, ftc.query_schema};
    auto join_operator         = std::make_shared<maximus::faiss::JoinExhaustiveOperator>(
        ftc.context, std::move(input_schemas), properties);
    join_operator->next_op_type     = maximus::PhysicalOperatorType::TABLE_SINK;
    join_operator->next_engine_type = maximus::EngineType::NATIVE;
    SchemaPtr result_schema         = std::make_shared<maximus::Schema>(
        arrow::FieldVector{arrow::field("id", arrow::int32()),
                           arrow::field("category", arrow::utf8()),
                           arrow::field("distance", arrow::utf8())});
    //std::cout << "operator = \n" << join_operator->to_string() << std::endl;

    // PUSH THE BATCH TO THE OPERATOR
    join_operator->add_input(maximus::DeviceTablePtr(ftc.query_table), 1);
    join_operator->add_input(maximus::DeviceTablePtr(ftc.data_table), 0);
    join_operator->no_more_input(0);
    join_operator->no_more_input(1);
    maximus::TablePtr output = join_operator->export_table();
}

TEST(faiss, JoinOperatorWithInnerProduct) {
    // CREATE OPERATOR
    auto properties = std::make_shared<maximus::VectorJoinExhaustiveProperties>(
        arrow::FieldRef("dvector"),
        arrow::FieldRef("qvector"),
        2,  // k = 2,
        std::nullopt,
        maximus::VectorDistanceMetric::INNER_PRODUCT);
    SchemaVector input_schemas = {ftc.data_schema, ftc.query_schema};
    auto join_operator         = std::make_shared<maximus::faiss::JoinExhaustiveOperator>(
        ftc.context, std::move(input_schemas), properties);
    join_operator->next_op_type     = maximus::PhysicalOperatorType::TABLE_SINK;
    join_operator->next_engine_type = maximus::EngineType::NATIVE;
    SchemaPtr result_schema         = std::make_shared<maximus::Schema>(arrow::FieldVector{
        arrow::field("id", arrow::int32()), arrow::field("category", arrow::utf8())});
    //std::cout << "operator = \n" << join_operator->to_string() << std::endl;

    // PUSH THE BATCH TO THE OPERATOR
    join_operator->add_input(maximus::DeviceTablePtr(ftc.query_table), 1);
    join_operator->add_input(maximus::DeviceTablePtr(ftc.data_table), 0);
    join_operator->no_more_input(0);
    join_operator->no_more_input(1);
    maximus::TablePtr output = join_operator->export_table();
    //output->print();
}

TEST(faiss, JoinOperatorRange) {
    // CREATE OPERATOR
    auto properties = std::make_shared<maximus::VectorJoinExhaustiveProperties>(
        arrow::FieldRef("dvector"),
        arrow::FieldRef("qvector"),
        std::nullopt,
        50.0,
        maximus::VectorDistanceMetric::INNER_PRODUCT);
    SchemaVector input_schemas = {ftc.data_schema, ftc.query_schema};
    auto join_operator         = std::make_shared<maximus::faiss::JoinExhaustiveOperator>(
        ftc.context, std::move(input_schemas), properties);
    join_operator->next_op_type     = maximus::PhysicalOperatorType::TABLE_SINK;
    join_operator->next_engine_type = maximus::EngineType::NATIVE;
    SchemaPtr result_schema         = std::make_shared<maximus::Schema>(arrow::FieldVector{
        arrow::field("id", arrow::int32()), arrow::field("category", arrow::utf8())});
    //std::cout << "operator = \n" << join_operator->to_string() << std::endl;

    // PUSH THE BATCH TO THE OPERATOR
    join_operator->add_input(maximus::DeviceTablePtr(ftc.query_table), 1);
    join_operator->add_input(maximus::DeviceTablePtr(ftc.data_table), 0);
    join_operator->no_more_input(0);
    join_operator->no_more_input(1);
    maximus::TablePtr output = join_operator->export_table();
    EXPECT_STREQ(output->to_string().data(),
                 "id(int32), dcategory(utf8)\n"
                 "1, c\n"
                 "1, d\n");
}

TEST(faiss, ProjectDistanceOperator) {
    // ===============================================
    //     CREATING THE INPUT
    // ===============================================
    arrow::FieldVector fields = {
        arrow::field("dvector", maximus::embeddings_list(arrow::float32(), 5)),  //  a 5d emb
        arrow::field("category", arrow::utf8())                                  //  a string
    };
    arrow::FieldVector query_fields = {
        arrow::field("id", arrow::int32()),
        arrow::field("qvector", maximus::embeddings_list(arrow::float32(), 5))  //  a 5d emb
    };
    auto input_schema  = std::make_shared<maximus::Schema>(fields);
    auto query_schema  = std::make_shared<maximus::Schema>(query_fields);
    auto result_schema = std::make_shared<maximus::Schema>(
        arrow::FieldVector{arrow::field("id", arrow::int32()),
                           arrow::field("category", arrow::utf8()),
                           arrow::field("qvDistance", arrow::float32())});

    auto context = maximus::make_context();
    maximus::TableBatchPtr table;
    maximus::TableBatchPtr query_vector;

    auto status = maximus::TableBatch::from_json(context,
                                                 input_schema,
                                                 {R"([
            [[1.00, 1.00, 1.00, 1.00, 1.00], "a"],
            [[1.00, 3.00, 1.00, 3.00, 1.00], "b"],
            [[2.00, 2.00, 8.00, 8.00, 8.00], "c"]
        ])"},
                                                 table);
    CHECK_STATUS(status);
    status = maximus::TableBatch::from_json(context,
                                            query_schema,
                                            {R"([
            [0, [0.99, 1.01, 0.98, 1.02, 0.99]],
            [1, [2.50, 1.50, 7.50, 8.50, 9.00]]
        ])"},
                                            query_vector);
    CHECK_STATUS(status);

    // ===============================================
    //     CREATING THE OPERATOR
    // ===============================================
    auto properties = std::make_shared<maximus::VectorProjectDistanceProperties>(
        "qvDistance", arrow::FieldRef("qvector"), arrow::FieldRef("dvector"));
    std::vector<std::shared_ptr<maximus::Schema>> input_schemas = {std::move(query_schema),
                                                                   std::move(input_schema)};

    auto pd_operator = std::make_shared<maximus::faiss::ProjectDistanceOperator>(
        context, std::move(input_schemas), properties);
    pd_operator->next_op_type     = maximus::PhysicalOperatorType::TABLE_SINK;
    pd_operator->next_engine_type = maximus::EngineType::NATIVE;
    //std::cout << "operator = \n" << pd_operator->to_string() << std::endl;

    // ===============================================
    //     PUSH THE BATCH TO THE OPERATOR
    // ===============================================
    pd_operator->add_input(maximus::DeviceTablePtr(std::move(table)), 1);
    pd_operator->add_input(maximus::DeviceTablePtr(std::move(query_vector)), 0);
    pd_operator->no_more_input(0);
    pd_operator->no_more_input(1);

    maximus::DeviceTablePtr output = pd_operator->export_next_batch();
    output.convert_to<maximus::TablePtr>(context, result_schema);
    std::string result = output.as_table()->to_string();
    //output_batch.as_table()->print();
    EXPECT_STREQ(result.data(),
                 "category(utf8), id(int32), qvDistance(float)\n"
                 "a, 0, 0.001101\n"
                 "b, 0, 7.881102\n"
                 "c, 0, 149.141098\n"
                 "a, 1, 165.000000\n"
                 "b, 1, 141.000000\n"
                 "c, 1, 2.000000\n");
}

TEST(faiss, KeepVectorColumnFlags) {
    // Test that keep_data_vector_column and keep_query_vector_column flags work correctly
    // for both JoinExhaustiveOperator and JoinIndexedOperator across all 4 combinations.
    // We don't have to provide functionality for dropping vectors cols as it can be a project
    // operator afterwards, but it's a very common use case so it's easiest this way
    // ===============================================
    //     CREATING THE INPUT
    // ===============================================
    auto data_schema = std::make_shared<maximus::Schema>(
        arrow::FieldVector{arrow::field("dvector", maximus::embeddings_list(arrow::float32(), 5)),
                           arrow::field("category", arrow::utf8()),
                           arrow::field("data_id", arrow::int32())});
    auto query_schema = std::make_shared<maximus::Schema>(
        arrow::FieldVector{arrow::field("qid", arrow::int32()),
                           arrow::field("qvector", maximus::embeddings_list(arrow::float32(), 5)),
                           arrow::field("query_label", arrow::utf8())});

    auto maximus_context = maximus::make_context();

    // Helper to create fresh copies of input tables for each test case
    auto make_data_table = [&]() {
        maximus::TableBatchPtr table;
        auto status = maximus::TableBatch::from_json(maximus_context,
                                                     data_schema,
                                                     {R"([
                [[1.00, 1.00, 1.00, 1.00, 1.00], "a", 100],
                [[2.00, 2.00, 2.00, 2.00, 2.00], "b", 200]
            ])"},
                                                     table);
        CHECK_STATUS(status);
        return table;
    };

    auto make_query_table = [&]() {
        maximus::TableBatchPtr table;
        auto status = maximus::TableBatch::from_json(maximus_context,
                                                     query_schema,
                                                     {R"([
                [0, [1.01, 0.99, 1.00, 1.02, 0.98], "query_0"]
            ])"},
                                                     table);
        CHECK_STATUS(status);
        return table;
    };

    // Build a simple flat index for indexed operator tests
    const int d              = 5;
    std::string index_string = "Flat";
    auto vector_index        = maximus::faiss::FaissIndex::factory_make(
        maximus_context, d, index_string, maximus::faiss::MetricType::METRIC_L2);
    auto sample_data = make_data_table();
    auto column      = sample_data->get_table_batch()->GetColumnByName("dvector");
    auto vectors     = std::static_pointer_cast<maximus::EmbeddingsArray>(column);
    // train should do nothing for "Flat" index, but call it anyway to test. If it fails you need to handle it.
    vector_index->train(*vectors);
    vector_index->add(*vectors);
    auto index_params = std::make_shared<maximus::faiss::FaissSearchParameters>(
        std::make_shared<::faiss::SearchParameters>());
    // Test configurations: (keep_data_vector, keep_query_vector)
    std::vector<std::pair<bool, bool>> all_cases = {
        {false, false}, {true, false}, {false, true}, {true, true}};

    for (auto [keep_dvec, keep_qvec] : all_cases) {
        SCOPED_TRACE(testing::Message()
                     << "Testing config: keep_dvec=" << keep_dvec << ", keep_qvec=" << keep_qvec);

        // ===============================================
        //     TEST INDEXED OPERATOR
        // ===============================================
        {
            auto data_table  = make_data_table();
            auto query_table = make_query_table();

            auto properties =
                std::make_shared<maximus::VectorJoinIndexedProperties>(arrow::FieldRef("dvector"),
                                                                       arrow::FieldRef("qvector"),
                                                                       vector_index,
                                                                       1,  // k = 1
                                                                       std::nullopt,
                                                                       index_params,
                                                                       keep_dvec,
                                                                       keep_qvec);

            std::vector<std::shared_ptr<maximus::Schema>> input_schemas = {data_schema,
                                                                           query_schema};

            auto join_operator = std::make_shared<maximus::faiss::JoinIndexedOperator>(
                maximus_context, std::move(input_schemas), properties);
            join_operator->next_op_type     = maximus::PhysicalOperatorType::TABLE_SINK;
            join_operator->next_engine_type = maximus::EngineType::NATIVE;

            join_operator->add_input(maximus::DeviceTablePtr(std::move(data_table)), 0);
            join_operator->add_input(maximus::DeviceTablePtr(std::move(query_table)), 1);
            join_operator->no_more_input(0);
            join_operator->no_more_input(1);

            maximus::DeviceTablePtr output = join_operator->export_next_batch();
            // Convert to Table to inspect results
            auto arrow_table   = output.as_arrow_table();
            auto output_schema = arrow_table->schema();

            // 1. Input columns besides the vectors should always be present
            std::vector<std::string> required_cols = {"qid", "query_label", "category", "data_id"};
            for (const auto& col : required_cols) {
                EXPECT_NE(output_schema->GetFieldByName(col), nullptr)
                    << "Missing required column: " << col;
            }

            // 2. Check the conditional columns (keep specific logic separate)
            EXPECT_EQ(output_schema->GetFieldByName("dvector") != nullptr, keep_dvec);
            EXPECT_EQ(output_schema->GetFieldByName("qvector") != nullptr, keep_qvec);
        }

        // ===============================================
        //     TEST EXHAUSTIVE OPERATOR
        // ===============================================
        {
            auto data_table  = make_data_table();
            auto query_table = make_query_table();

            auto properties = std::make_shared<maximus::VectorJoinExhaustiveProperties>(
                arrow::FieldRef("dvector"),
                arrow::FieldRef("qvector"),
                1,             // k = 1
                std::nullopt,  // radius none
                maximus::VectorDistanceMetric::L2,
                keep_dvec,
                keep_qvec);

            std::vector<std::shared_ptr<maximus::Schema>> input_schemas = {data_schema,
                                                                           query_schema};

            auto join_operator = std::make_shared<maximus::faiss::JoinExhaustiveOperator>(
                maximus_context, std::move(input_schemas), properties);
            join_operator->next_op_type     = maximus::PhysicalOperatorType::TABLE_SINK;
            join_operator->next_engine_type = maximus::EngineType::NATIVE;

            join_operator->add_input(maximus::DeviceTablePtr(std::move(data_table)), 0);
            join_operator->add_input(maximus::DeviceTablePtr(std::move(query_table)), 1);
            join_operator->no_more_input(0);
            join_operator->no_more_input(1);

            maximus::DeviceTablePtr output = join_operator->export_next_batch();
            // Convert to Table to inspect results
            auto arrow_table   = output.as_arrow_table();
            auto output_schema = arrow_table->schema();

            // 1. Input columns besides the vectors should always be present
            std::vector<std::string> required_cols = {"qid", "query_label", "category", "data_id"};
            for (const auto& col : required_cols) {
                EXPECT_NE(output_schema->GetFieldByName(col), nullptr)
                    << "Missing required column: " << col;
            }

            // 2. Check the conditional columns (keep specific logic separate)
            EXPECT_EQ(output_schema->GetFieldByName("dvector") != nullptr, keep_dvec);
            EXPECT_EQ(output_schema->GetFieldByName("qvector") != nullptr, keep_qvec);
        }
    }
}

TEST(faiss_interop, RawPointerExtractionWithMixedColumns) {
    // 1. Define Common Data (qid, vector, category)
    //    3 rows, vector dimension = 3
    std::string json_data = R"([
        [101, [1.1, 1.2, 1.3], "cat_A"], 
        [102, [2.1, 2.2, 2.3], "cat_B"], 
        [103, [3.1, 3.2, 3.3], "cat_A"],
        [104, [4.1, 4.2, 4.3], "cat_C"],
        [105, [5.1, 5.2, 5.3], "cat_B"]
    ])";

    int D = 3;
    // 2. Define Two Schemas
    // Schema A: Variable List (maximus::embeddings_list -> arrow::ListArray (maximus::EmbeddingsArray))
    auto schema_list = std::make_shared<maximus::Schema>(
        arrow::FieldVector{arrow::field("qid", arrow::int32()),
                           arrow::field("vector", maximus::embeddings_list(arrow::float32(), D)),
                           arrow::field("category", arrow::utf8())});

    // Schema B: Fixed Size List (arrow::fixed_size_list -> arrow::FixedSizeListArray)
    auto schema_fixed = std::make_shared<maximus::Schema>(
        arrow::FieldVector{arrow::field("qid", arrow::int32()),
                           arrow::field("vector", arrow::fixed_size_list(arrow::float32(), D)),
                           arrow::field("category", arrow::utf8())});

    // 3. Load Data into Tables
    auto context = maximus::make_context();
    maximus::TableBatchPtr table_list;
    maximus::TableBatchPtr table_fixed;

    CHECK_STATUS(maximus::TableBatch::from_json(context, schema_list, {json_data}, table_list));
    CHECK_STATUS(maximus::TableBatch::from_json(context, schema_fixed, {json_data}, table_fixed));

    // Get the vector columns specifically
    // Note: We cast them to their respective Arrow types immediately
    auto col_list = std::static_pointer_cast<maximus::EmbeddingsArray>(
        table_list->get_table_batch()->GetColumnByName("vector"));
    auto col_fixed = std::static_pointer_cast<arrow::FixedSizeListArray>(
        table_fixed->get_table_batch()->GetColumnByName("vector"));

    // 4. Test Loop: Iterate over SLICES
    // We simulate processing the data in small batches (chunks)
    int64_t total_rows = 5;
    int64_t chunk_size = 2;

    for (int64_t start = 0; start < total_rows; start += chunk_size) {
        int64_t length = std::min(chunk_size, total_rows - start);

        // CREATE SLICES (Views)
        auto slice_list =
            std::static_pointer_cast<maximus::EmbeddingsArray>(col_list->Slice(start, length));
        auto slice_fixed =
            std::static_pointer_cast<arrow::FixedSizeListArray>(col_fixed->Slice(start, length));

        std::cout << "Testing Slice: Start=" << start << ", Length=" << length << std::endl;

        // EXTRACT POINTERS
        float* ptr_list  = maximus::raw_ptr_from_array(*slice_list);
        float* ptr_fixed = maximus::raw_ptr_from_array(*slice_fixed);

        ASSERT_NE(ptr_list, nullptr);
        ASSERT_NE(ptr_fixed, nullptr);

        // Verify the pointers point to identical values
        for (int i = 0; i < length * D; ++i) {
            float val_list  = ptr_list[i];
            float val_fixed = ptr_fixed[i];

            EXPECT_FLOAT_EQ(val_list, val_fixed)
                << "Mismatch at local float index " << i << " of slice starting at row " << start;
        }

        // Optional: Prove we got the correct row by checking the value
        // Row 0 starts with 1.1, Row 1 with 2.1... Row N with (N+1).1
        float expected_first_val = (float) (start + 1) + 0.1f;
        EXPECT_NEAR(ptr_list[0], expected_first_val, 0.0001);

        std::cout << "  Verified Slice[" << start << "] starts with: " << ptr_list[0] << std::endl;
    }
}


TEST_F(Sift10KTest, IndexCacheSaveLoadCPU) {
    // Build an index with caching enabled, then rebuild/load it and
    // verify both indices produce identical search results.

    auto training_data    = db->get_table("base");
    bool use_cache        = true;
    std::string cache_dir = "/tmp/index_cache";
    // std::string desc      = "IVF256,PQ8";
    std::string desc = "IVF256,Flat";
    // std::string desc      = "HNSW32,Flat";


    std::shared_ptr<maximus::faiss::FaissIndex> index1;
    index1 = maximus::faiss::FaissIndex::build(context,
                                               training_data,
                                               "vector",
                                               desc,
                                               maximus::VectorDistanceMetric::L2,
                                               use_cache,
                                               cache_dir);

    ASSERT_NE(index1, nullptr);
    EXPECT_EQ(index1->device_type, maximus::DeviceType::CPU);

    std::shared_ptr<maximus::faiss::FaissIndex> index2;
    index2 = maximus::faiss::FaissIndex::build(context,
                                               training_data,
                                               "vector",
                                               desc,
                                               maximus::VectorDistanceMetric::L2,
                                               use_cache,
                                               cache_dir);

    ASSERT_NE(index2, nullptr);
    EXPECT_EQ(index2->device_type, maximus::DeviceType::CPU);

    std::vector<float> distances1(nq * k);
    std::vector<faiss::idx_t> labels1(nq * k);

    std::vector<float> distances2(nq * k);
    std::vector<faiss::idx_t> labels2(nq * k);

    index1->faiss_index->search(nq, query_vectors.data(), k, distances1.data(), labels1.data());
    index2->faiss_index->search(nq, query_vectors.data(), k, distances2.data(), labels2.data());

    for (size_t i = 0; i < labels1.size(); ++i) {
        EXPECT_EQ(labels1[i], labels2[i]) << "mismatch at index " << i;
        EXPECT_NEAR(distances1[i], distances2[i], 1e-6) << "distance mismatch at index " << i;
    }

    //Cleanup cache directory
    std::filesystem::remove_all(cache_dir);
    std::cout << "Cleaned up cache directory: " << cache_dir << std::endl;
}


TEST_F(Sift10KTest, WrapperVsRawSearch) {
    // Test that searching via the Maximus FaissIndex wrapper
    // produces identical results to searching via the raw FAISS index.

    auto training_data = db->get_table("base");
    auto ctx           = db->get_context();

    bool use_cache        = false;
    std::string cache_dir = "./index_cache_test_wrapper_raw";
    std::string desc      = "HNSW32,Flat";

    auto index = maximus::faiss::FaissIndex::build(ctx,
                                                   training_data,
                                                   "vector",
                                                   desc,
                                                   maximus::VectorDistanceMetric::L2,
                                                   use_cache,
                                                   cache_dir);
    ASSERT_NE(index, nullptr);

    auto faiss_index = std::dynamic_pointer_cast<maximus::faiss::FaissIndex>(index);
    ASSERT_NE(faiss_index, nullptr);

    // Build an Arrow query array for the first query vector from the fixture.
    // The fixture exposes `query_vectors` (flat float buffer), `nq` and `k`.
    ASSERT_GT(nq, 0);
    int d = static_cast<int>(query_vectors.size() / nq);
    ASSERT_GT(d, 0);

    // Create a small Arrow table with a single query (id + vector) using the fixture context
    auto schema = std::make_shared<maximus::Schema>(
        arrow::FieldVector{arrow::field("id", arrow::int32()),
                           arrow::field("vector", maximus::embeddings_list(arrow::float32(), d))});

    // Use the fixture-provided Arrow array and take the first query
    auto query_vectors_arr =
        std::static_pointer_cast<maximus::EmbeddingsArray>(query_array->Slice(0, 1));

    int k = 2;  // compare top-2

    // Wrapper search (fills these backing vectors)
    std::vector<float> distances_w(k);
    std::vector<int64_t> labels_w(k);
    auto dist_array  = std::make_shared<arrow::FloatArray>(k, arrow::Buffer::Wrap(distances_w));
    auto label_array = std::make_shared<arrow::Int64Array>(k, arrow::Buffer::Wrap(labels_w));

    index->search(*query_vectors_arr, k, *dist_array, *label_array);

    // Raw FAISS search
    std::vector<float> distances_r(k);
    std::vector<faiss::idx_t> labels_r(k);
    float* xq = maximus::raw_ptr_from_array(*query_vectors_arr);
    ASSERT_NE(xq, nullptr);
    faiss_index->faiss_index->search(1, xq, k, distances_r.data(), labels_r.data());

    // Compare results
    for (int i = 0; i < k; ++i) {
        EXPECT_EQ(labels_w[i], static_cast<int64_t>(labels_r[i])) << "label mismatch at " << i;
        EXPECT_NEAR(distances_w[i], distances_r[i], 1e-5) << "distance mismatch at " << i;
    }
}


TEST_F(Sift10KTest, FrontendIndexedVectorJoinCPU) {
    auto ctx = db->get_context();

    // Get training data for index
    auto training_data = db->get_table("base");

    // Create query plan using the new overload
    auto data_source =
        table_source(db, "base", data_schema, {"id", "vector"}, maximus::DeviceType::CPU);
    auto query_source =
        table_source(db, "query", query_schema, {"id", "vector"}, maximus::DeviceType::CPU);

    auto data_renamed  = rename(data_source, {"id", "vector"}, {"data_id", "vector"});
    auto query_renamed = rename(query_source, {"id", "vector"}, {"query_id", "vector"});

    // NOTE: the build should be done "offline" before the query node is created
    bool use_cache        = false;
    std::string cache_dir = "./index_cache";
    auto index            = maximus::faiss::FaissIndex::build(ctx,
                                                   training_data,
                                                   "vector",
                                                   "Flat",
                                                   maximus::VectorDistanceMetric::L2,
                                                   use_cache,
                                                   cache_dir);

    // 2) Create the index outside the query node (in theory "offline" )
    auto join_node = maximus::indexed_vector_join(data_renamed,
                                                  query_renamed,
                                                  "vector",      // data vector column name
                                                  "vector",      // query vector column name
                                                  index,         // <--- PASS THE BUILT INDEX HERE
                                                  10,            // K = 10
                                                  std::nullopt,  // no radius
                                                  nullptr,       // no search params
                                                  false,         // don't keep data vector
                                                  false,         // don't keep query vector
                                                  "distance",    // distance column name
                                                  maximus::DeviceType::CPU);

    auto qp = query_plan(table_sink(join_node));
    db->schedule(qp);
    ctx->barrier();
    auto tables = db->execute();
    ctx->barrier();

    ASSERT_EQ(tables.size(), 1);
    auto result = tables[0];

    // Should have nq queries * 10 neighbors = 1000 rows (nq = 100 for SIFT10K query set)
    EXPECT_EQ(result->num_rows(), nq * k);

    // if you want to print result
    // result->print();

    // Verify distance column exists and has valid values
    auto labels_col = result->get_table()->GetColumnByName("data_id");
    ASSERT_NE(labels_col, nullptr);

    auto dist_col = result->get_table()->GetColumnByName("distance");
    ASSERT_NE(dist_col, nullptr);

    // Coalesce chunked columns into single arrays (handles multi-chunk results)
    std::shared_ptr<arrow::Int64Array> labels_array;
    // TODO: Fix impot, add arrow helper cast from 32 to 64 array or accept 32 arraw in the VerifyArrowResults
    // If the column is int32 (common for parquet), cast to int64 using Arrow compute
    if (labels_col->type()->id() == arrow::Type::INT32) {
        auto tmp32 = maximus::arrow_array<arrow::Int32Array>(labels_col);
        arrow::Result<arrow::Datum> cast_res =
            arrow::compute::Cast(arrow::Datum(tmp32), arrow::int64());
        ASSERT_TRUE(cast_res.ok()) << cast_res.status().ToString();
        labels_array =
            std::static_pointer_cast<arrow::Int64Array>(cast_res.ValueOrDie().make_array());
    } else {
        labels_array = maximus::arrow_array<arrow::Int64Array>(labels_col);
    }
    // auto labels_array = maximus::arrow_array<arrow::Int64Array>(labels_col);
    auto dist_array = maximus::arrow_array<arrow::FloatArray>(dist_col);

    // Verify distances are non-negative and exact results for GPU Flat
    VerifyArrowResults(*labels_array, *dist_array, /*is_exact=*/true, "Flat");

    // Or if you plan to verify recall manually with std::vectr
    // Option 2: You need a) concat to single chunk and b) copy to std::vector c) use the recall verifier
    // - note that VerifyArrowResults already does a,b,c but in a different way. This is just for illustration:
    // But this second way is faster as it's a single copy, instead of per-element access in VerifyArrowResults

    // BRIDGE CODE: Arrow Arrays -> std::vectors (for recall)
    const float* raw_dists = dist_array->raw_values();
    const int64_t* raw_ids = labels_array->raw_values();
    std::vector<float> distances(raw_dists, raw_dists + dist_array->length());
    std::vector<::faiss::idx_t> labels(raw_ids, raw_ids + labels_array->length());

    // Verify distances are sorted and results are reasonable
    VerifyDistancesSorted(distances, k);
    // GPU Flat should give 100% recall (exact search)
    VerifyExactResults(labels, k);
    // TODO: Later, not now I will fix Recall tests
}

TEST(faiss, LargeListArrayBuildCPU) {
    // Test building a CPU index with LargeListArray (int64 offsets)
    auto large_list_schema = std::make_shared<maximus::Schema>(arrow::FieldVector{
        arrow::field("vector", arrow::large_list(arrow::float32())),
        arrow::field("label", arrow::utf8())});

    auto context = maximus::make_context();
    maximus::TableBatchPtr table;
    auto status = maximus::TableBatch::from_json(context,
                                                 large_list_schema,
                                                 {R"([
            [[1.0, 2.0, 3.0, 4.0, 5.0], "a"],
            [[2.0, 3.0, 4.0, 5.0, 6.0], "b"],
            [[3.0, 4.0, 5.0, 6.0, 7.0], "c"]
        ])"},
                                                 table);
    CHECK_STATUS(status);

    // Build CPU index from DeviceTablePtr
    maximus::DeviceTablePtr data_table(table);
    auto index = maximus::faiss::FaissIndex::build(context,
                                                   data_table,
                                                   "vector",
                                                   "Flat",
                                                   maximus::VectorDistanceMetric::L2,
                                                   false);
    ASSERT_NE(index, nullptr);
    EXPECT_EQ(index->device_type, maximus::DeviceType::CPU);
    EXPECT_EQ(index->faiss_index->ntotal, 3);
}

}  // namespace test
