#include <gtest/gtest.h>

#include <cudf/copying.hpp>
#include <cudf/interop.hpp>
#include <cudf/join/join.hpp>
#include <cudf/stream_compaction.hpp>
#include <fstream>
#include <maximus/context.hpp>
#include <maximus/dag/query_plan.hpp>
#include <maximus/gpu/cuda_api.hpp>
#include <maximus/gpu/cudf/interop.hpp>
#include <maximus/operators/gpu/cudf/distinct_operator.cpp>
#include <maximus/operators/gpu/cudf/filter_operator.hpp>
#include <maximus/operators/gpu/cudf/group_by_operator.hpp>
#include <maximus/operators/gpu/cudf/hash_join_operator.hpp>
#include <maximus/operators/gpu/cudf/limit_operator.cpp>
#include <maximus/operators/gpu/cudf/local_broadcast_operator.hpp>
#include <maximus/operators/gpu/cudf/order_by_operator.hpp>
#include <maximus/operators/gpu/cudf/project_operator.hpp>
#include <maximus/operators/native/random_table_source_operator.hpp>
#include <maximus/types/aggregate.hpp>
#include <maximus/types/table_batch.hpp>
#include <maximus/utils/arrow_helpers.hpp>
#include <maximus/utils/json_helpers.hpp>
#include <typeinfo>

namespace test {

TEST(DataTransfer, Cuda) {
    // Generate a new schema
    arrow::FieldVector fields;
    int num_rows = 400;
    fields.emplace_back(std::make_shared<arrow::Field>("test1", arrow::int64()));
    fields.emplace_back(std::make_shared<arrow::Field>("test2", arrow::boolean()));
    fields.emplace_back(std::make_shared<arrow::Field>("test3", arrow::utf8()));
    auto schema = std::make_shared<maximus::Schema>(std::make_shared<arrow::Schema>(fields));

    // Generate new recordbatch
    std::shared_ptr<arrow::RecordBatch> batch = maximus::generate_batch(fields, num_rows, 0);

    // Transfer recordbatch to GPU GTable
    auto ctx = std::make_shared<maximus::MaximusContext>();
    maximus::DeviceTablePtr device_ptr(batch);
    device_ptr.to_gpu(ctx, schema);
    std::shared_ptr<maximus::gpu::GTable> tab1 = device_ptr.as_gpu();

    // Clone GTable
    std::shared_ptr<maximus::gpu::GTable> tab2 = tab1->clone();

    // Transfer GPU GTables back to new recordbatch
    maximus::DeviceTablePtr device_ptr1(tab1);
    device_ptr1.convert_to<maximus::ArrowTableBatchPtr>(ctx, schema);
    std::shared_ptr<arrow::RecordBatch> fin1 = device_ptr1.as_arrow_table_batch();

    maximus::DeviceTablePtr device_ptr2(tab2);
    device_ptr2.convert_to<maximus::ArrowTableBatchPtr>(ctx, schema);
    std::shared_ptr<arrow::RecordBatch> fin2 = device_ptr2.as_arrow_table_batch();
    // std::cout << batch->ToString() << std::endl;
    // std::cout << fin->ToString() << std::endl;
    ASSERT_TRUE(fin1->Equals(*batch));
    ASSERT_TRUE(fin2->Equals(*batch));
}

TEST(DataTransfer, Cudf) {
    // Generate a new schema
    arrow::FieldVector fields;
    int num_rows = 500;
    fields.emplace_back(std::make_shared<arrow::Field>("test1", arrow::date32()));
    fields.emplace_back(std::make_shared<arrow::Field>("test3", arrow::utf8()));
    fields.emplace_back(
        std::make_shared<arrow::Field>("test4", arrow::timestamp(arrow::TimeUnit::SECOND)));
    auto schema = std::make_shared<maximus::Schema>(std::make_shared<arrow::Schema>(fields));

    // Generate new recordbatch
    std::shared_ptr<arrow::RecordBatch> batch = maximus::generate_batch(fields, num_rows, 0);

    // Transfer recordbatch to GPU GTable
    auto ctx = std::make_shared<maximus::MaximusContext>();
    maximus::DeviceTablePtr device_ptr(batch);
    device_ptr.to_gpu(ctx, schema);
    std::shared_ptr<maximus::gpu::GTable> tab = device_ptr.as_gpu();

    // Read GTable as cudf table
    std::shared_ptr<::cudf::table> cudf_table = maximus::gpu::gtable_to_cudf(tab);

    // Write cudf table back to GTable
    std::shared_ptr<maximus::gpu::GTable> tab2 =
        maximus::gpu::cudf_to_gtable(ctx, schema, cudf_table);

    // Transfer GPU GTable back to new recordbatch
    maximus::DeviceTablePtr device_ptr2(std::move(tab2));
    device_ptr2.convert_to<maximus::ArrowTableBatchPtr>(ctx, schema);
    std::shared_ptr<arrow::RecordBatch> fin = device_ptr2.as_arrow_table_batch();
    // std::cout << batch->ToString() << std::endl;
    // std::cout << fin->ToString() << std::endl;
    ASSERT_TRUE(fin->Equals(*batch));
}

TEST(DataConversion, ArrowToCudf) {
    // Generate a new schema
    arrow::FieldVector fields;
    int num_rows = 500;
    fields.emplace_back(std::make_shared<arrow::Field>("test1", arrow::date32()));
    fields.emplace_back(std::make_shared<arrow::Field>("test3", arrow::utf8()));
    fields.emplace_back(
        std::make_shared<arrow::Field>("test4", arrow::timestamp(arrow::TimeUnit::SECOND)));

    auto schema = std::make_shared<maximus::Schema>(std::make_shared<arrow::Schema>(fields));

    // Generate new recordbatch
    std::shared_ptr<arrow::RecordBatch> batch = maximus::generate_batch(fields, num_rows, 0);

    auto ctx      = maximus::make_context();
    auto pool     = ctx->get_memory_pool();
    auto stream   = ::cudf::get_default_stream();
    auto gpu_pool = rmm::mr::get_current_device_resource();

    std::cout << "converting Arrow -> Cudf" << std::endl;
    std::unique_ptr<cudf::table> cudf_table =
        cudf_from_arrow_batch(batch, schema, stream, gpu_pool, pool);

    std::cout << "converting Cudf -> Arrow" << std::endl;
    std::shared_ptr<arrow::RecordBatch> batch_out =
        cudf_to_arrow_batch(cudf_table->view(), schema, stream, gpu_pool, pool);

    std::cout << batch->ToString() << std::endl;
    std::cout << batch_out->ToString() << std::endl;
    ASSERT_TRUE(batch->Equals(*batch_out));
}

maximus::TableBatchPtr to_cpu(maximus::DeviceTablePtr batch,
                              std::shared_ptr<maximus::MaximusContext> &context,
                              std::shared_ptr<maximus::Schema> schema) {
    // copy to CPU
    if (!batch.on_cpu()) {
        batch.to_cpu(context, schema);
    }
    // cast to TableBatchPtr
    return batch.as_cpu();
}

TEST(CudfGroupBy, MultiTable) {
    // ===============================================
    //     CREATING THE INPUT SCHEMA
    // ===============================================
    // a source node generating random tuples
    auto fields = {
        arrow::field("key1", arrow::utf8()),   //  a string
        arrow::field("key2", arrow::int8()),   //  a bool
        arrow::field("value", arrow::int32())  // an integer
    };
    auto input_schema = std::make_shared<maximus::Schema>(fields);

    // ===============================================
    //     EXPECTED OUTPUT SCHEMA
    // ===============================================
    auto expected_fields = {
        arrow::field("key1", arrow::utf8()),        // a string
        arrow::field("key2", arrow::int8()),        //  a bool
        arrow::field("value_sum", arrow::int64()),  // an integer
        arrow::field("value_min", arrow::int32()),  // an integer
        arrow::field("value_max", arrow::int32())   // an integer
    };
    auto output_schema = std::make_shared<maximus::Schema>(std::move(expected_fields));

    // ===============================================
    //     CREATING THE AGGREGATES
    // ===============================================
    auto arrow_aggregate1 =
             std::make_shared<arrow::compute::Aggregate>("hash_sum", "value", "value_sum"),
         arrow_aggregate2 =
             std::make_shared<arrow::compute::Aggregate>("hash_min", "value", "value_min"),
         arrow_aggregate3 =
             std::make_shared<arrow::compute::Aggregate>("hash_max", "value", "value_max");

    auto aggregate1 = std::make_shared<maximus::Aggregate>(std::move(arrow_aggregate1)),
         aggregate2 = std::make_shared<maximus::Aggregate>(std::move(arrow_aggregate2)),
         aggregate3 = std::make_shared<maximus::Aggregate>(std::move(arrow_aggregate3));

    // ===============================================
    //     CREATING THE GROUP KEYS
    // ===============================================
    std::vector<arrow::FieldRef> group_keys                     = {"key1", "key2"};
    std::vector<std::shared_ptr<maximus::Aggregate>> aggregates = {
        std::move(aggregate1), std::move(aggregate2), std::move(aggregate3)};
    // std::cout << "Group keys[0] = " << group_keys[0] << std::endl;

    // ===============================================
    //     WRAPPING INSIDE THE GROUP BY PROPERTIES
    // ===============================================
    auto group_by_properties =
        std::make_shared<maximus::GroupByProperties>(std::move(group_keys), std::move(aggregates));
    // std::cout << "Group by properties = \n" <<
    // group_by_properties->to_string() << std::endl;

    // ===============================================
    //     CREATING THE OPERATOR
    // ===============================================
    // auto gcontext = maximus::gpu::make_cuda_context();
    auto context = maximus::make_context();
    // context->gcontext = std::move(gcontext);
    auto group_by_operator = std::make_shared<maximus::cudf::GroupByOperator>(
        context, input_schema, std::move(group_by_properties));

    // std::cout << "Finished creating the operator" << std::endl;
    // std::cout << "operator = \n" << group_by_operator->to_string() <<
    // std::endl;

    // ===============================================
    //     GENERATE A RANDOM INPUT BATCH
    // ===============================================
    maximus::TableBatchPtr batch1, batch2;
    auto status = maximus::TableBatch::from_json(context,
                                                 input_schema,
                                                 {R"([
            ["x", 1, 1],
            ["x", 1, 2],
            ["y", 0, 3],
            ["y", 0, 4],
            ["z", 1, 5]
        ])"},
                                                 batch1);
    CHECK_STATUS(status);
    status = maximus::TableBatch::from_json(context,
                                            input_schema,
                                            {R"([
            ["x", 1, 1],
            ["x", 1, 2],
            ["y", 0, 3],
            ["y", 0, 4],
            ["z", 1, 5]
        ])"},
                                            batch2);
    CHECK_STATUS(status);
    std::cout << "===========================" << std::endl;
    std::cout << "     The input batch1 =     " << std::endl;
    std::cout << "===========================" << std::endl;
    batch1->print();
    std::cout << "===========================" << std::endl;
    std::cout << "     The input batch2 =     " << std::endl;
    std::cout << "===========================" << std::endl;
    batch2->print();

    // ===============================================
    //     PUSH THE BATCH TO THE OPERATOR
    // ===============================================

    group_by_operator->add_input(maximus::DeviceTablePtr(std::move(batch1)), 0);

    group_by_operator->add_input(maximus::DeviceTablePtr(std::move(batch2)), 0);

    group_by_operator->no_more_input(0);

    // ===============================================
    //     PRINT THE OUTPUT BATCHES
    // ===============================================
    int num_batches = 0;
    maximus::TableBatchPtr output_batch;
    while (group_by_operator->has_more_batches(true)) {
        output_batch = to_cpu(group_by_operator->export_next_batch(), context, output_schema);
        num_batches++;
    }
    EXPECT_EQ(num_batches, 1);

    // ===============================================
    //     EXPECTED OUTPUT BATCH
    // ===============================================
    maximus::TableBatchPtr expected;
    status = maximus::TableBatch::from_json(context,
                                            output_schema,
                                            {R"([
            ["x", 1, 6, 1, 2],
            ["z", 1, 10, 5, 5],
            ["y", 0, 14, 3, 4]
        ])"},
                                            expected);

    CHECK_STATUS(status);
    std::cout << "===========================" << std::endl;
    std::cout << "     The output batch      " << std::endl;
    std::cout << "===========================" << std::endl;
    output_batch->print();
    std::cout << "===========================" << std::endl;
    std::cout << " The expected output batch " << std::endl;
    std::cout << "===========================" << std::endl;
    expected->print();

    EXPECT_TRUE(*output_batch == *expected);
}

TEST(CudfOrderBy, MultiTable) {
    // ===============================================
    //     CREATING THE INPUT SCHEMA
    // ===============================================
    // a source node generating random tuples
    auto fields = {
        arrow::field("key1", arrow::utf8()),   //  a string
        arrow::field("key2", arrow::int32()),  //  an integer
        arrow::field("value", arrow::int32())  // an integer
    };
    auto input_schema = std::make_shared<maximus::Schema>(fields);

    // ===============================================
    //     EXPECTED OUTPUT SCHEMA
    // ===============================================
    auto expected_fields = {
        arrow::field("key1", arrow::utf8()),   //  a string
        arrow::field("key2", arrow::int32()),  //  an integer
        arrow::field("value", arrow::int32())  // an integer
    };
    auto output_schema = std::make_shared<maximus::Schema>(std::move(expected_fields));

    // ===============================================
    //     CREATING THE SORT KEYS
    // ===============================================
    std::vector<maximus::SortKey> sort_keys = {{"key1", maximus::SortOrder::ASCENDING},
                                               {"key2", maximus::SortOrder::DESCENDING}};

    maximus::NullOrder null_order = maximus::NullOrder::FIRST;

    // ===============================================
    //     WRAPPING INSIDE THE GROUP BY PROPERTIES
    // ===============================================
    auto order_by_properties =
        std::make_shared<maximus::OrderByProperties>(std::move(sort_keys), null_order);
    // std::cout << "Order by properties = \n"
    //           << order_by_properties->to_string() << std::endl;

    // ===============================================
    //     CREATING THE OPERATOR
    // ===============================================
    // auto gcontext = maximus::gpu::make_cuda_context();
    auto context = maximus::make_context();
    // context->gcontext = std::move(gcontext);
    auto order_by_operator = std::make_shared<maximus::cudf::OrderByOperator>(
        context, input_schema, std::move(order_by_properties));

    std::cout << "Finished creating the operator" << std::endl;
    std::cout << "operator = \n" << order_by_operator->to_string() << std::endl;

    // ===============================================
    //     GENERATE A RANDOM INPUT BATCH
    // ===============================================
    maximus::TableBatchPtr batch1, batch2;
    auto status = maximus::TableBatch::from_json(context,
                                                 input_schema,
                                                 {R"([
            ["y", 4, 4],
            ["x", 2, 1],
            ["y", 2, 3],
            ["x", 3, 2],
            ["z", 5, 5]
        ])"},
                                                 batch1);
    status      = maximus::TableBatch::from_json(context,
                                            input_schema,
                                                 {R"([
            ["y", 7, 4],
            ["x", 5, 1],
            ["y", 6, 3],
            ["x", 5, 2],
            ["z", 7, 5]
        ])"},
                                            batch2);

    CHECK_STATUS(status);
    std::cout << "===========================" << std::endl;
    std::cout << "     The input batch =     " << std::endl;
    std::cout << "===========================" << std::endl;
    batch1->print();
    std::cout << "===========================" << std::endl;
    std::cout << "     The input batch =     " << std::endl;
    std::cout << "===========================" << std::endl;
    batch2->print();

    // ===============================================
    //     PUSH THE BATCH TO THE OPERATOR
    // ===============================================

    order_by_operator->add_input(maximus::DeviceTablePtr(std::move(batch1)), 0);

    order_by_operator->add_input(maximus::DeviceTablePtr(std::move(batch2)), 0);

    order_by_operator->no_more_input(0);

    // ===============================================
    //     PRINT THE OUTPUT BATCHES
    // ===============================================
    int num_batches = 0;
    maximus::TableBatchPtr output_batch;
    while (order_by_operator->has_more_batches(true)) {
        output_batch = to_cpu(order_by_operator->export_next_batch(), context, output_schema);
        num_batches++;
    }
    EXPECT_EQ(num_batches, 1);

    // ===============================================
    //     EXPECTED OUTPUT BATCH
    // ===============================================
    maximus::TableBatchPtr expected;
    status = maximus::TableBatch::from_json(context,
                                            output_schema,
                                            {R"([
            ["x", 5, 1],
            ["x", 5, 2],
            ["x", 3, 2],
            ["x", 2, 1],
            ["y", 7, 4],
            ["y", 6, 3],
            ["y", 4, 4],
            ["y", 2, 3],
            ["z", 7, 5],
            ["z", 5, 5]
        ])"},
                                            expected);

    CHECK_STATUS(status);
    std::cout << "===========================" << std::endl;
    std::cout << "     The output batch      " << std::endl;
    std::cout << "===========================" << std::endl;
    output_batch->print();
    std::cout << "===========================" << std::endl;
    std::cout << " The expected output batch " << std::endl;
    std::cout << "===========================" << std::endl;
    expected->print();

    EXPECT_TRUE(*output_batch == *expected);
}

TEST(CudfHashJoin, MultiTable) {
    // ===============================================
    //     CREATING THE INPUT SCHEMAS
    // ===============================================
    // a source node generating random tuples
    auto fields1 = {
        arrow::field("key1", arrow::int8()),  // an integer
        arrow::field("key2", arrow::int8()),  // an integer
        arrow::field("value", arrow::int8())  // an integer
    };
    auto input_schema1 = std::make_shared<maximus::Schema>(fields1);

    auto fields2 = {
        arrow::field("key1", arrow::int8()),  // an integer
        arrow::field("key2", arrow::int8()),  // an integer
        arrow::field("value", arrow::int8())  // an integer
    };
    auto input_schema2 = std::make_shared<maximus::Schema>(fields2);

    // ===============================================
    //     EXPECTED OUTPUT SCHEMA
    // ===============================================
    auto expected_fields = {
        arrow::field("key1_l", arrow::int8()),   // an integer
        arrow::field("key2_l", arrow::int8()),   // an integer
        arrow::field("value_l", arrow::int8()),  // an integer
        arrow::field("key1_r", arrow::int8()),   // an integer
        arrow::field("key2_r", arrow::int8()),   // an integer
        arrow::field("value_r", arrow::int8()),  // an integer
    };
    auto output_schema = std::make_shared<maximus::Schema>(std::move(expected_fields));

    // ===============================================
    //     CREATING THE JOIN KEYS
    // ===============================================
    std::vector<arrow::FieldRef> left_keys  = {arrow::FieldRef("key1"), arrow::FieldRef("key2")},
                                 right_keys = {arrow::FieldRef("key1"), arrow::FieldRef("key2")};

    // ===============================================
    //     WRAPPING INSIDE THE JOIN PROPERTIES
    // ===============================================
    auto join_properties = std::make_shared<maximus::JoinProperties>(
        maximus::JoinType::INNER,
        std::move(left_keys),
        std::move(right_keys),
        std::make_shared<maximus::Expression>(
            std::make_shared<arrow::compute::Expression>(arrow::compute::literal(true))),
        "_l",
        "_r");

    // std::cout << "Join properties = \n"
    //           << join_properties->to_string() << std::endl;

    // ===============================================
    //     CREATING THE OPERATOR
    // ===============================================
    // auto gcontext = maximus::gpu::make_cuda_context();
    auto context = maximus::make_context();
    // context->gcontext = std::move(gcontext);
    auto hash_join_operator = std::make_shared<maximus::cudf::HashJoinOperator>(
        context, input_schema1, input_schema2, std::move(join_properties));

    // std::cout << "Finished creating the operator" << std::endl;
    // std::cout << "operator = \n" << hash_join_operator->to_string() <<
    // std::endl;

    // ===============================================
    //     GENERATE A RANDOM INPUT BATCH
    // ===============================================
    maximus::TableBatchPtr batch1[2], batch2[3];
    auto status = maximus::TableBatch::from_json(context,
                                                 input_schema1,
                                                 {R"([
            [0, 0, 1]
        ])"},
                                                 batch1[0]);
    CHECK_STATUS(status);
    std::cout << "===========================" << std::endl;
    std::cout << "   The input batch 1.0 =   " << std::endl;
    std::cout << "===========================" << std::endl;
    batch1[0]->print();
    status = maximus::TableBatch::from_json(context,
                                            input_schema1,
                                            {R"([
            [1, 1, 3],
            [2, 4, 5]
        ])"},
                                            batch1[1]);
    CHECK_STATUS(status);
    std::cout << "===========================" << std::endl;
    std::cout << "   The input batch 1.1 =   " << std::endl;
    std::cout << "===========================" << std::endl;
    batch1[1]->print();

    status = maximus::TableBatch::from_json(context,
                                            input_schema1,
                                            {R"([
            [0, 0, 11]
        ])"},
                                            batch2[0]);
    CHECK_STATUS(status);
    std::cout << "===========================" << std::endl;
    std::cout << "   The input batch 2.0 =   " << std::endl;
    std::cout << "===========================" << std::endl;
    batch2[0]->print();
    status = maximus::TableBatch::from_json(context,
                                            input_schema1,
                                            {R"([
            [0, 3, 13]
        ])"},
                                            batch2[1]);
    CHECK_STATUS(status);
    std::cout << "===========================" << std::endl;
    std::cout << "   The input batch 2.1 =   " << std::endl;
    std::cout << "===========================" << std::endl;
    batch2[1]->print();
    status = maximus::TableBatch::from_json(context,
                                            input_schema1,
                                            {R"([
            [2, 4, 17]
        ])"},
                                            batch2[2]);
    CHECK_STATUS(status);
    std::cout << "===========================" << std::endl;
    std::cout << "   The input batch 2.2 =   " << std::endl;
    std::cout << "===========================" << std::endl;
    batch2[2]->print();

    // ===============================================
    //     PUSH THE BATCH TO THE OPERATOR
    // ===============================================

    hash_join_operator->add_input(maximus::DeviceTablePtr(std::move(batch1[0])), 0);

    hash_join_operator->add_input(maximus::DeviceTablePtr(std::move(batch2[0])), 1);

    hash_join_operator->add_input(maximus::DeviceTablePtr(std::move(batch1[1])), 0);

    hash_join_operator->add_input(maximus::DeviceTablePtr(std::move(batch2[1])), 1);

    hash_join_operator->add_input(maximus::DeviceTablePtr(std::move(batch2[2])), 1);

    hash_join_operator->no_more_input(0);

    hash_join_operator->no_more_input(1);

    // ===============================================
    //     PRINT THE OUTPUT BATCHES
    // ===============================================
    int num_batches = 0;
    maximus::TableBatchPtr output_batch;
    while (hash_join_operator->has_more_batches(true)) {
        output_batch = to_cpu(hash_join_operator->export_next_batch(), context, output_schema);
        num_batches++;
    }
    EXPECT_EQ(num_batches, 1);

    // ===============================================
    //     EXPECTED OUTPUT BATCH
    // ===============================================
    maximus::TableBatchPtr expected;
    status = maximus::TableBatch::from_json(context,
                                            output_schema,
                                            {R"([
            [0, 0, 1, 0, 0, 11],
            [2, 4, 5, 2, 4, 17]
        ])"},
                                            expected);

    CHECK_STATUS(status);
    std::cout << "===========================" << std::endl;
    std::cout << "     The output batch      " << std::endl;
    std::cout << "===========================" << std::endl;
    output_batch->print();
    std::cout << "===========================" << std::endl;
    std::cout << " The expected output batch " << std::endl;
    std::cout << "===========================" << std::endl;
    expected->print();

    EXPECT_TRUE(*output_batch == *expected);
}

TEST(CudfProject, MultiTest) {
    // ===============================================
    //     CREATING THE INPUT SCHEMAS
    // ===============================================
    // a source node generating random tuples
    auto fields1 = {
        arrow::field("a", arrow::int8()),  // an integer
        arrow::field("b", arrow::int8()),  // an integer
        arrow::field("c", arrow::int8())   // an integer
    };
    auto input_schema = std::make_shared<maximus::Schema>(fields1);

    // ===============================================
    //     EXPECTED OUTPUT SCHEMA
    // ===============================================
    auto expected_fields = {
        arrow::field("ab", arrow::int32()),  // an integer
        arrow::field("bc", arrow::int8()),   // an integer
        arrow::field("ca", arrow::int8()),   // an integer
    };
    auto output_schema = std::make_shared<maximus::Schema>(std::move(expected_fields));

    // ===============================================
    //     CREATING THE PROJECT EXPRESSIONS
    // ===============================================
    arrow::compute::Expression expr1 = arrow::compute::call("bit_wise_or",
                                                            {arrow::compute::field_ref("a"),
                                                             arrow::compute::field_ref("b")}),
                               expr2 = arrow::compute::field_ref("b"),
                               expr3 = arrow::compute::field_ref("c");
    std::shared_ptr<arrow::compute::Expression>
        pexpr1 = std::make_shared<arrow::compute::Expression>(expr1),
        pexpr2 = std::make_shared<arrow::compute::Expression>(expr2),
        pexpr3 = std::make_shared<arrow::compute::Expression>(expr3);
    std::vector<std::shared_ptr<maximus::Expression>> exprs = {
        std::make_shared<maximus::Expression>(pexpr1),
        std::make_shared<maximus::Expression>(pexpr2),
        std::make_shared<maximus::Expression>(pexpr3)};

    // ===============================================
    //     WRAPPING INSIDE THE PROJECT PROPERTIES
    // ===============================================
    auto project_properties = std::make_shared<maximus::ProjectProperties>(std::move(exprs));
    project_properties->column_names = {"ab", "bc", "ca"};

    // std::cout << "Project properties = \n"
    //           << project_properties->to_string() << std::endl;

    // ===============================================
    //     CREATING THE OPERATOR
    // ===============================================
    // auto gcontext = maximus::gpu::make_cuda_context();
    auto context = maximus::make_context();
    // context->gcontext = std::move(gcontext);
    auto project_operator = std::make_shared<maximus::cudf::ProjectOperator>(
        context, input_schema, std::move(project_properties));

    // std::cout << "Finished creating the operator" << std::endl;
    // std::cout << "operator = \n" << project_operator->to_string() <<
    // std::endl;

    // ===============================================
    //     GENERATE A RANDOM INPUT BATCH
    // ===============================================
    maximus::TableBatchPtr batch1, batch2;
    auto status = maximus::TableBatch::from_json(context,
                                                 input_schema,
                                                 {R"([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1]
        ])"},
                                                 batch1);
    CHECK_STATUS(status);
    std::cout << "===========================" << std::endl;
    std::cout << "    The input batch 1  =   " << std::endl;
    std::cout << "===========================" << std::endl;
    batch1->print();
    status = maximus::TableBatch::from_json(context,
                                            input_schema,
                                            {R"([
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1]
        ])"},
                                            batch2);
    CHECK_STATUS(status);
    std::cout << "===========================" << std::endl;
    std::cout << "   The input batch 2 =    " << std::endl;
    std::cout << "===========================" << std::endl;
    batch2->print();

    // ===============================================
    //     PUSH THE BATCH TO THE OPERATOR
    // ===============================================

    project_operator->add_input(maximus::DeviceTablePtr(std::move(batch1)), 0);

    project_operator->add_input(maximus::DeviceTablePtr(std::move(batch2)), 0);

    project_operator->no_more_input(0);

    // ===============================================
    //     PRINT THE OUTPUT BATCHES
    // ===============================================
    int num_batches = 0;
    maximus::TableBatchPtr output_batch1, output_batch2;
    while (project_operator->has_more_batches(false)) {
        num_batches == 0
            ? output_batch1 = to_cpu(project_operator->export_next_batch(), context, output_schema)
            : output_batch2 = to_cpu(project_operator->export_next_batch(), context, output_schema);
        num_batches++;
    }
    EXPECT_EQ(num_batches, 2);

    // ===============================================
    //     EXPECTED OUTPUT BATCH
    // ===============================================
    maximus::TableBatchPtr expected1, expected2;
    status = maximus::TableBatch::from_json(context,
                                            output_schema,
                                            {R"([
            [0, 0, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 1, 1]
        ])"},
                                            expected1);
    CHECK_STATUS(status);

    status = maximus::TableBatch::from_json(context,
                                            output_schema,
                                            {R"([
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1]
        ])"},
                                            expected2);
    CHECK_STATUS(status);

    std::cout << "===========================" << std::endl;
    std::cout << "     The output batch 1    " << std::endl;
    std::cout << "===========================" << std::endl;
    output_batch1->print();
    std::cout << "===========================" << std::endl;
    std::cout << "     The output batch 2    " << std::endl;
    std::cout << "===========================" << std::endl;
    output_batch2->print();

    std::cout << "===========================" << std::endl;
    std::cout << "The expected output batch 1" << std::endl;
    std::cout << "===========================" << std::endl;
    expected1->print();
    std::cout << "===========================" << std::endl;
    std::cout << "The expected output batch 2" << std::endl;
    std::cout << "===========================" << std::endl;
    expected2->print();

    EXPECT_TRUE(*output_batch1 == *expected1);
    EXPECT_TRUE(*output_batch2 == *expected2);
}

TEST(CudfProject, LargeStringTest) {
    // ===============================================
    //     CREATING THE INPUT SCHEMAS
    // ===============================================
    // a source node generating random tuples
    auto fields1 = {
        arrow::field("a", arrow::utf8()),  // a string
        arrow::field("b", arrow::int8()),  // an integer
        arrow::field("c", arrow::int8())   // an integer
    };
    auto input_schema = std::make_shared<maximus::Schema>(fields1);

    // ===============================================
    //     EXPECTED OUTPUT SCHEMA
    // ===============================================
    auto expected_fields = {
        arrow::field("a", arrow::utf8()),    // an integer
        arrow::field("bc", arrow::int32()),  // an integer
    };
    auto output_schema = std::make_shared<maximus::Schema>(std::move(expected_fields));

    // ===============================================
    //     CREATING THE PROJECT EXPRESSIONS
    // ===============================================
    std::shared_ptr<arrow::compute::FunctionOptions> opt =
        std::make_shared<arrow::compute::SliceOptions>(0, 2);
    arrow::compute::Expression expr1 = arrow::compute::call(
                                   "utf8_slice_codeunits", {arrow::compute::field_ref("a")}, opt),
                               expr2 = arrow::compute::call("bit_wise_or",
                                                            {arrow::compute::field_ref("b"),
                                                             arrow::compute::field_ref("c")});
    std::shared_ptr<arrow::compute::Expression> pexpr1 =
                                                    std::make_shared<arrow::compute::Expression>(
                                                        expr1),
                                                pexpr2 =
                                                    std::make_shared<arrow::compute::Expression>(
                                                        expr2);
    std::vector<std::shared_ptr<maximus::Expression>> exprs = {
        std::make_shared<maximus::Expression>(pexpr1),
        std::make_shared<maximus::Expression>(pexpr2)};

    // ===============================================
    //     WRAPPING INSIDE THE PROJECT PROPERTIES
    // ===============================================
    auto project_properties = std::make_shared<maximus::ProjectProperties>(std::move(exprs));
    project_properties->column_names = {"a", "bc"};

    // std::cout << "Project properties = \n"
    //           << project_properties->to_string() << std::endl;

    // ===============================================
    //     CREATING THE OPERATOR
    // ===============================================
    // auto gcontext = maximus::gpu::make_cuda_context();
    auto context = maximus::make_context();
    // context->gcontext = std::move(gcontext);
    auto project_operator = std::make_shared<maximus::cudf::ProjectOperator>(
        context, input_schema, std::move(project_properties));

    // std::cout << "Finished creating the operator" << std::endl;
    // std::cout << "operator = \n" << project_operator->to_string() <<
    // std::endl;

    // ===============================================
    //     GENERATE A RANDOM INPUT BATCH
    // ===============================================
    maximus::TableBatchPtr batch1, batch2;
    auto status = maximus::TableBatch::from_json(context,
                                                 input_schema,
                                                 {R"([
            ["abc", 0, 0],
            ["abd", 0, 1],
            ["abe", 1, 0],
            ["abf", 1, 1]
        ])"},
                                                 batch1);
    CHECK_STATUS(status);
    std::cout << "===========================" << std::endl;
    std::cout << "    The input batch 1  =   " << std::endl;
    std::cout << "===========================" << std::endl;
    batch1->print();
    status = maximus::TableBatch::from_json(context,
                                            input_schema,
                                            {R"([
            ["zbc", 0, 0],
            ["zbd", 0, 1],
            ["zbe", 1, 0],
            ["zbf", 1, 1]
        ])"},
                                            batch2);
    CHECK_STATUS(status);
    std::cout << "===========================" << std::endl;
    std::cout << "   The input batch 2 =    " << std::endl;
    std::cout << "===========================" << std::endl;
    batch2->print();

    // ===============================================
    //     PUSH THE BATCH TO THE OPERATOR
    // ===============================================

    project_operator->add_input(maximus::DeviceTablePtr(std::move(batch1)), 0);

    project_operator->add_input(maximus::DeviceTablePtr(std::move(batch2)), 0);

    project_operator->no_more_input(0);

    // ===============================================
    //     PRINT THE OUTPUT BATCHES
    // ===============================================
    int num_batches = 0;
    maximus::TableBatchPtr output_batch1, output_batch2;
    while (project_operator->has_more_batches(false)) {
        num_batches == 0
            ? output_batch1 = to_cpu(project_operator->export_next_batch(), context, output_schema)
            : output_batch2 = to_cpu(project_operator->export_next_batch(), context, output_schema);
        num_batches++;
    }
    EXPECT_EQ(num_batches, 2);

    // ===============================================
    //     EXPECTED OUTPUT BATCH
    // ===============================================
    maximus::TableBatchPtr expected1, expected2;
    status = maximus::TableBatch::from_json(context,
                                            output_schema,
                                            {R"([
            ["ab", 0],
            ["ab", 1],
            ["ab", 1],
            ["ab", 1]
        ])"},
                                            expected1);
    CHECK_STATUS(status);

    status = maximus::TableBatch::from_json(context,
                                            output_schema,
                                            {R"([
            ["zb", 0],
            ["zb", 1],
            ["zb", 1],
            ["zb", 1]
        ])"},
                                            expected2);
    CHECK_STATUS(status);

    std::cout << "===========================" << std::endl;
    std::cout << "     The output batch 1    " << std::endl;
    std::cout << "===========================" << std::endl;
    output_batch1->print();
    std::cout << "===========================" << std::endl;
    std::cout << "     The output batch 2    " << std::endl;
    std::cout << "===========================" << std::endl;
    output_batch2->print();

    std::cout << "===========================" << std::endl;
    std::cout << "The expected output batch 1" << std::endl;
    std::cout << "===========================" << std::endl;
    expected1->print();
    std::cout << "===========================" << std::endl;
    std::cout << "The expected output batch 2" << std::endl;
    std::cout << "===========================" << std::endl;
    expected2->print();

    EXPECT_TRUE(*output_batch1 == *expected1);
    EXPECT_TRUE(*output_batch2 == *expected2);
}

TEST(CudfProject, IfTest) {
    // ===============================================
    //     CREATING THE INPUT SCHEMAS
    // ===============================================
    // a source node generating random tuples
    auto fields1 = {
        arrow::field("a", arrow::int8()),  // an integer
        arrow::field("b", arrow::int8()),  // an integer
        arrow::field("c", arrow::int8())   // an integer
    };
    auto input_schema = std::make_shared<maximus::Schema>(fields1);

    // ===============================================
    //     EXPECTED OUTPUT SCHEMA
    // ===============================================
    auto expected_fields = {
        arrow::field("ifathenbelsec", arrow::int32())  // an integer
    };
    auto output_schema = std::make_shared<maximus::Schema>(std::move(expected_fields));

    // ===============================================
    //     CREATING THE PROJECT EXPRESSIONS
    // ===============================================
    arrow::compute::Expression if_expr = arrow::compute::call(
        "if_else",
        {arrow::compute::call(
             "equal", {arrow::compute::field_ref("a"), arrow::compute::literal((int8_t) 0)}),
         arrow::compute::field_ref("b"),
         arrow::compute::field_ref("c")});
    arrow::compute::Expression expr2 = arrow::compute::call(
        "subtract",
        {arrow::compute::literal((int8_t) 1),
         arrow::compute::call(
             "if_else",
             {arrow::compute::call("equal", {if_expr, arrow::compute::literal((int8_t) 1)}),
              arrow::compute::literal((int8_t) 0),
              arrow::compute::literal((int8_t) 1)})});
    std::shared_ptr<arrow::compute::Expression> pexpr =
        std::make_shared<arrow::compute::Expression>(expr2);
    std::vector<std::shared_ptr<maximus::Expression>> exprs = {
        std::make_shared<maximus::Expression>(pexpr)};

    // ===============================================
    //     WRAPPING INSIDE THE PROJECT PROPERTIES
    // ===============================================
    auto project_properties = std::make_shared<maximus::ProjectProperties>(std::move(exprs));
    project_properties->column_names = {"ifathenbelsec"};

    // std::cout << "Project properties = \n"
    //           << project_properties->to_string() << std::endl;

    // ===============================================
    //     CREATING THE OPERATOR
    // ===============================================
    // auto gcontext = maximus::gpu::make_cuda_context();
    auto context = maximus::make_context();
    // context->gcontext = std::move(gcontext);
    auto project_operator = std::make_shared<maximus::cudf::ProjectOperator>(
        context, input_schema, std::move(project_properties));

    // std::cout << "Finished creating the operator" << std::endl;
    // std::cout << "operator = \n" << project_operator->to_string() <<
    // std::endl;

    // ===============================================
    //     GENERATE A RANDOM INPUT BATCH
    // ===============================================
    maximus::TableBatchPtr batch;
    auto status = maximus::TableBatch::from_json(context,
                                                 input_schema,
                                                 {R"([
            [0, 0, 1],
            [0, 1, 0],
            [1, 1, 0],
            [1, 0, 1]
        ])"},
                                                 batch);
    CHECK_STATUS(status);
    std::cout << "===========================" << std::endl;
    std::cout << "     The input batch  =    " << std::endl;
    std::cout << "===========================" << std::endl;
    batch->print();

    // ===============================================
    //     PUSH THE BATCH TO THE OPERATOR
    // ===============================================

    project_operator->add_input(maximus::DeviceTablePtr(std::move(batch)), 0);

    project_operator->no_more_input(0);

    // ===============================================
    //     PRINT THE OUTPUT BATCHES
    // ===============================================
    int num_batches = 0;
    maximus::TableBatchPtr output_batch;
    while (project_operator->has_more_batches(false)) {
        output_batch = to_cpu(project_operator->export_next_batch(), context, output_schema);
        num_batches++;
    }
    EXPECT_EQ(num_batches, 1);

    // ===============================================
    //     EXPECTED OUTPUT BATCH
    // ===============================================
    maximus::TableBatchPtr expected;
    status = maximus::TableBatch::from_json(context,
                                            output_schema,
                                            {R"([
            [0],
            [1],
            [0],
            [1]
        ])"},
                                            expected);
    CHECK_STATUS(status);

    std::cout << "===========================" << std::endl;
    std::cout << "      The output batch     " << std::endl;
    std::cout << "===========================" << std::endl;
    output_batch->print();

    std::cout << "===========================" << std::endl;
    std::cout << " The expected output batch " << std::endl;
    std::cout << "===========================" << std::endl;
    expected->print();

    EXPECT_TRUE(*output_batch == *expected);
}

TEST(CudfProject, MoveTest) {
    // ===============================================
    //     CREATING THE INPUT SCHEMAS
    // ===============================================
    // a source node generating random tuples
    auto fields1 = {
        arrow::field("a", arrow::int8()),  // an integer
        arrow::field("b", arrow::int8()),  // an integer
        arrow::field("c", arrow::int8())   // an integer
    };
    auto input_schema = std::make_shared<maximus::Schema>(fields1);

    // ===============================================
    //     EXPECTED OUTPUT SCHEMA
    // ===============================================
    auto expected_fields = {
        arrow::field("d", arrow::int8()),  // an integer
        arrow::field("e", arrow::int8()),  // an integer
    };
    auto output_schema = std::make_shared<maximus::Schema>(std::move(expected_fields));

    // ===============================================
    //     CREATING THE PROJECT EXPRESSIONS
    // ===============================================
    arrow::compute::Expression expr1 = arrow::compute::field_ref("a"),
                               expr2 = arrow::compute::field_ref("b");
    std::shared_ptr<arrow::compute::Expression> pexpr1 =
                                                    std::make_shared<arrow::compute::Expression>(
                                                        expr1),
                                                pexpr2 =
                                                    std::make_shared<arrow::compute::Expression>(
                                                        expr2);
    std::vector<std::shared_ptr<maximus::Expression>> exprs = {
        std::make_shared<maximus::Expression>(pexpr1),
        std::make_shared<maximus::Expression>(pexpr2)};

    // ===============================================
    //     WRAPPING INSIDE THE PROJECT PROPERTIES
    // ===============================================
    auto project_properties = std::make_shared<maximus::ProjectProperties>(std::move(exprs));
    project_properties->column_names = {"d", "e"};

    // std::cout << "Project properties = \n"
    //           << project_properties->to_string() << std::endl;

    // ===============================================
    //     CREATING THE OPERATOR
    // ===============================================
    // auto gcontext = maximus::gpu::make_cuda_context();
    auto context = maximus::make_context();
    // context->gcontext = std::move(gcontext);
    auto project_operator = std::make_shared<maximus::cudf::ProjectOperator>(
        context, input_schema, std::move(project_properties));

    // std::cout << "Finished creating the operator" << std::endl;
    // std::cout << "operator = \n" << project_operator->to_string() <<
    // std::endl;

    // ===============================================
    //     GENERATE A RANDOM INPUT BATCH
    // ===============================================
    maximus::TableBatchPtr batch1, batch2;
    auto status = maximus::TableBatch::from_json(context,
                                                 input_schema,
                                                 {R"([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1]
        ])"},
                                                 batch1);
    CHECK_STATUS(status);
    std::cout << "===========================" << std::endl;
    std::cout << "    The input batch 1  =   " << std::endl;
    std::cout << "===========================" << std::endl;
    batch1->print();
    status = maximus::TableBatch::from_json(context,
                                            input_schema,
                                            {R"([
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1]
        ])"},
                                            batch2);
    CHECK_STATUS(status);
    std::cout << "===========================" << std::endl;
    std::cout << "   The input batch 2 =    " << std::endl;
    std::cout << "===========================" << std::endl;
    batch2->print();

    // ===============================================
    //     PUSH THE BATCH TO THE OPERATOR
    // ===============================================

    project_operator->add_input(maximus::DeviceTablePtr(std::move(batch1)), 0);

    project_operator->add_input(maximus::DeviceTablePtr(std::move(batch2)), 0);

    project_operator->no_more_input(0);

    // ===============================================
    //     PRINT THE OUTPUT BATCHES
    // ===============================================
    int num_batches = 0;
    maximus::TableBatchPtr output_batch1, output_batch2;
    while (project_operator->has_more_batches(false)) {
        num_batches == 0
            ? output_batch1 = to_cpu(project_operator->export_next_batch(), context, output_schema)
            : output_batch2 = to_cpu(project_operator->export_next_batch(), context, output_schema);
        num_batches++;
    }
    EXPECT_EQ(num_batches, 2);

    // ===============================================
    //     EXPECTED OUTPUT BATCH
    // ===============================================
    maximus::TableBatchPtr expected1, expected2;
    status = maximus::TableBatch::from_json(context,
                                            output_schema,
                                            {R"([
            [0, 0],
            [0, 0],
            [0, 1],
            [0, 1]
        ])"},
                                            expected1);
    CHECK_STATUS(status);

    status = maximus::TableBatch::from_json(context,
                                            output_schema,
                                            {R"([
            [1, 0],
            [1, 0],
            [1, 1],
            [1, 1]
        ])"},
                                            expected2);
    CHECK_STATUS(status);

    std::cout << "===========================" << std::endl;
    std::cout << "     The output batch 1    " << std::endl;
    std::cout << "===========================" << std::endl;
    output_batch1->print();
    std::cout << "===========================" << std::endl;
    std::cout << "     The output batch 2    " << std::endl;
    std::cout << "===========================" << std::endl;
    output_batch2->print();

    std::cout << "===========================" << std::endl;
    std::cout << "The expected output batch 1" << std::endl;
    std::cout << "===========================" << std::endl;
    expected1->print();
    std::cout << "===========================" << std::endl;
    std::cout << "The expected output batch 2" << std::endl;
    std::cout << "===========================" << std::endl;
    expected2->print();

    EXPECT_TRUE(*output_batch1 == *expected1);
    EXPECT_TRUE(*output_batch2 == *expected2);
}

TEST(CudfFilter, MultiTest) {
    // ===============================================
    //     CREATING THE INPUT SCHEMAS
    // ===============================================
    // a source node generating random tuples
    auto fields1 = {
        arrow::field("a", arrow::int8()),  // an integer
        arrow::field("b", arrow::int8()),  // an integer
        arrow::field("c", arrow::int8())   // an integer
    };
    auto input_schema = std::make_shared<maximus::Schema>(fields1);

    // ===============================================
    //     EXPECTED OUTPUT SCHEMA
    // ===============================================
    auto expected_fields = {
        arrow::field("a", arrow::int8()),  // an integer
        arrow::field("b", arrow::int8()),  // an integer
        arrow::field("c", arrow::int8())   // an integer
    };
    auto output_schema = std::make_shared<maximus::Schema>(std::move(expected_fields));

    // ===============================================
    //     CREATING THE FILTER EXPRESSIONS
    // ===============================================
    arrow::compute::Expression expr1 = arrow::compute::call(
        "greater", {arrow::compute::field_ref("a"), arrow::compute::literal((int8_t) 0)});
    std::shared_ptr<arrow::compute::Expression> expr =
        std::make_shared<arrow::compute::Expression>(expr1);
    std::shared_ptr<maximus::Expression> mexpr = std::make_shared<maximus::Expression>(expr);

    // ===============================================
    //     WRAPPING INSIDE THE FILTER PROPERTIES
    // ===============================================
    auto filter_properties = std::make_shared<maximus::FilterProperties>(std::move(mexpr));

    // std::cout << "Filter properties = \n"
    //           << filter_properties->to_string() << std::endl;

    // ===============================================
    //     CREATING THE OPERATOR
    // ===============================================
    // auto gcontext = maximus::gpu::make_cuda_context();
    auto context = maximus::make_context();
    // context->gcontext = std::move(gcontext);
    auto filter_operator = std::make_shared<maximus::cudf::FilterOperator>(
        context, input_schema, std::move(filter_properties));

    // std::cout << "Finished creating the operator" << std::endl;
    // std::cout << "operator = \n" << filter_operator->to_string() <<
    // std::endl;

    // ===============================================
    //     GENERATE A RANDOM INPUT BATCH
    // ===============================================
    maximus::TableBatchPtr batch1, batch2;
    auto status = maximus::TableBatch::from_json(context,
                                                 input_schema,
                                                 {R"([
            [1, 2, 3],
            [-1, -2, -3],
            [2, 3, 4],
            [-2, -3, -4]
        ])"},
                                                 batch1);
    CHECK_STATUS(status);
    std::cout << "===========================" << std::endl;
    std::cout << "    The input batch 1  =   " << std::endl;
    std::cout << "===========================" << std::endl;
    batch1->print();
    status = maximus::TableBatch::from_json(context,
                                            input_schema,
                                            {R"([
            [3, 4, 5],
            [-3, -4, -5],
            [4, 5, 6],
            [-4, -5, -6]
        ])"},
                                            batch2);
    CHECK_STATUS(status);
    std::cout << "===========================" << std::endl;
    std::cout << "   The input batch 2 =    " << std::endl;
    std::cout << "===========================" << std::endl;
    batch2->print();

    // ===============================================
    //     PUSH THE BATCH TO THE OPERATOR
    // ===============================================

    filter_operator->add_input(maximus::DeviceTablePtr(std::move(batch1)), 0);

    filter_operator->add_input(maximus::DeviceTablePtr(std::move(batch2)), 0);

    filter_operator->no_more_input(0);

    // ===============================================
    //     PRINT THE OUTPUT BATCHES
    // ===============================================
    int num_batches = 0;
    maximus::TableBatchPtr output_batch1, output_batch2;
    while (filter_operator->has_more_batches(false)) {
        num_batches == 0
            ? output_batch1 = to_cpu(filter_operator->export_next_batch(), context, output_schema)
            : output_batch2 = to_cpu(filter_operator->export_next_batch(), context, output_schema);
        num_batches++;
    }
    EXPECT_EQ(num_batches, 2);

    // ===============================================
    //     EXPECTED OUTPUT BATCH
    // ===============================================
    maximus::TableBatchPtr expected1, expected2;
    status = maximus::TableBatch::from_json(context,
                                            output_schema,
                                            {R"([
            [1, 2, 3],
            [2, 3, 4]
        ])"},
                                            expected1);
    CHECK_STATUS(status);

    status = maximus::TableBatch::from_json(context,
                                            output_schema,
                                            {R"([
            [3, 4, 5],
            [4, 5, 6]
        ])"},
                                            expected2);
    CHECK_STATUS(status);

    std::cout << "===========================" << std::endl;
    std::cout << "     The output batch 1    " << std::endl;
    std::cout << "===========================" << std::endl;
    output_batch1->print();
    std::cout << "===========================" << std::endl;
    std::cout << "     The output batch 2    " << std::endl;
    std::cout << "===========================" << std::endl;
    output_batch2->print();

    std::cout << "===========================" << std::endl;
    std::cout << "The expected output batch 1" << std::endl;
    std::cout << "===========================" << std::endl;
    expected1->print();
    std::cout << "===========================" << std::endl;
    std::cout << "The expected output batch 2" << std::endl;
    std::cout << "===========================" << std::endl;
    expected2->print();

    EXPECT_TRUE(*output_batch1 == *expected1);
    EXPECT_TRUE(*output_batch2 == *expected2);
}

TEST(CudfFilter, IfTest) {
    // ===============================================
    //     CREATING THE INPUT SCHEMAS
    // ===============================================
    // a source node generating random tuples
    auto fields1 = {
        arrow::field("a", arrow::utf8()),  // a string
        arrow::field("c", arrow::int8())   // an integer
    };
    auto input_schema = std::make_shared<maximus::Schema>(fields1);

    // ===============================================
    //     EXPECTED OUTPUT SCHEMA
    // ===============================================
    auto expected_fields = {
        arrow::field("a", arrow::utf8()),  //  a string
        arrow::field("c", arrow::int8())   // an integer
    };
    auto output_schema = std::make_shared<maximus::Schema>(std::move(expected_fields));

    // ===============================================
    //     CREATING THE FILTER EXPRESSIONS
    // ===============================================
    // arrow::compute::Expression if_expr = arrow::compute::call(
    //     "or",
    //     {arrow::compute::call("starts_with",
    //                           {arrow::compute::field_ref("a")},
    //                           std::make_shared<arrow::compute::MatchSubstringOptions>("a")),
    //      arrow::compute::call("ends_with",
    //                           {arrow::compute::field_ref("a")},
    //                           std::make_shared<arrow::compute::MatchSubstringOptions>("a"))});
    arrow::compute::Expression if_expr =
        arrow::compute::call("match_like",
                             {arrow::compute::field_ref("a")},
                             std::make_shared<arrow::compute::MatchSubstringOptions>("[.]*a[.]*"));
    std::shared_ptr<arrow::compute::Expression> pexpr =
        std::make_shared<arrow::compute::Expression>(if_expr);
    std::shared_ptr<maximus::Expression> mexpr = std::make_shared<maximus::Expression>(pexpr);

    // ===============================================
    //     WRAPPING INSIDE THE FILTER PROPERTIES
    // ===============================================
    auto filter_properties = std::make_shared<maximus::FilterProperties>(std::move(mexpr));

    // std::cout << "Filter properties = \n"
    //           << filter_properties->to_string() << std::endl;

    // ===============================================
    //     CREATING THE OPERATOR
    // ===============================================
    // auto gcontext = maximus::gpu::make_cuda_context();
    auto context = maximus::make_context();
    // context->gcontext = std::move(gcontext);
    auto filter_operator = std::make_shared<maximus::cudf::FilterOperator>(
        context, input_schema, std::move(filter_properties));

    // std::cout << "Finished creating the operator" << std::endl;
    // std::cout << "operator = \n" << filter_operator->to_string() <<
    // std::endl;

    // ===============================================
    //     GENERATE A RANDOM INPUT BATCH
    // ===============================================
    maximus::TableBatchPtr batch;
    auto status = maximus::TableBatch::from_json(context,
                                                 input_schema,
                                                 {R"([
            ["a a", 0],
            ["a b", 1],
            ["a c", 2],
            ["d a", 3],
            ["e a", 4],
            ["f g", 5],
            ["h i", 6]
        ])"},
                                                 batch);
    CHECK_STATUS(status);
    std::cout << "===========================" << std::endl;
    std::cout << "     The input batch  =    " << std::endl;
    std::cout << "===========================" << std::endl;
    batch->print();

    // ===============================================
    //     PUSH THE BATCH TO THE OPERATOR
    // ===============================================

    filter_operator->add_input(maximus::DeviceTablePtr(std::move(batch)), 0);

    filter_operator->no_more_input(0);

    // ===============================================
    //     PRINT THE OUTPUT BATCHES
    // ===============================================
    int num_batches = 0;
    maximus::TableBatchPtr output_batch;
    while (filter_operator->has_more_batches(false)) {
        output_batch = to_cpu(filter_operator->export_next_batch(), context, output_schema);
        num_batches++;
    }
    EXPECT_EQ(num_batches, 1);

    // ===============================================
    //     EXPECTED OUTPUT BATCH
    // ===============================================
    maximus::TableBatchPtr expected;
    status = maximus::TableBatch::from_json(context,
                                            output_schema,
                                            {R"([
            ["a a", 0],
            ["a b", 1],
            ["a c", 2],
            ["d a", 3],
            ["e a", 4]
        ])"},
                                            expected);
    CHECK_STATUS(status);

    std::cout << "===========================" << std::endl;
    std::cout << "      The output batch     " << std::endl;
    std::cout << "===========================" << std::endl;
    output_batch->print();

    std::cout << "===========================" << std::endl;
    std::cout << " The expected output batch " << std::endl;
    std::cout << "===========================" << std::endl;
    expected->print();

    EXPECT_TRUE(*output_batch == *expected);
}

TEST(CudfDistinct, MultiTable) {
    // ===============================================
    //     CREATING THE INPUT SCHEMA
    // ===============================================
    // a source node generating random tuples
    auto fields = {
        arrow::field("key1", arrow::utf8()),   //  a string
        arrow::field("key2", arrow::int8()),   //  a bool
        arrow::field("value", arrow::int32())  // an integer
    };
    auto input_schema = std::make_shared<maximus::Schema>(fields);

    // ===============================================
    //     EXPECTED OUTPUT SCHEMA
    // ===============================================
    auto expected_fields = {
        arrow::field("key1", arrow::utf8()),   //  a string
        arrow::field("key2", arrow::int8()),   //  a bool
        arrow::field("value", arrow::int32())  // an integer
    };
    auto output_schema = std::make_shared<maximus::Schema>(std::move(expected_fields));

    // ===============================================
    //     CREATING THE DISTINCT KEYS
    // ===============================================
    std::vector<arrow::FieldRef> distinct_keys = {"key1", "key2"};
    // std::cout << "Distinct keys[0] = " << distinct_keys[0] << std::endl;

    // ===============================================
    //     WRAPPING INSIDE THE GROUP BY PROPERTIES
    // ===============================================
    auto distinct_properties =
        std::make_shared<maximus::DistinctProperties>(std::move(distinct_keys));
    // std::cout << "Distinct properties = \n" <<
    // distinct_properties->to_string() << std::endl;

    // ===============================================
    //     CREATING THE OPERATOR
    // ===============================================
    // auto gcontext = maximus::gpu::make_cuda_context();
    auto context = maximus::make_context();
    // context->gcontext = std::move(gcontext);
    auto distinct_operator = std::make_shared<maximus::cudf::DistinctOperator>(
        context, input_schema, std::move(distinct_properties));

    // std::cout << "Finished creating the operator" << std::endl;
    // std::cout << "operator = \n" << distinct_operator->to_string() <<
    // std::endl;

    // ===============================================
    //     GENERATE A RANDOM INPUT BATCH
    // ===============================================
    maximus::TableBatchPtr batch1, batch2;
    auto status = maximus::TableBatch::from_json(context,
                                                 input_schema,
                                                 {R"([
            ["x", 1, 1],
            ["x", 2, 2],
            ["y", 0, 3],
            ["y", 1, 4],
            ["z", 1, 5]
        ])"},
                                                 batch1);
    CHECK_STATUS(status);
    status = maximus::TableBatch::from_json(context,
                                            input_schema,
                                            {R"([
            ["x", 1, 1],
            ["x", 3, 2],
            ["y", 0, 3],
            ["y", 2, 4],
            ["z", 1, 5]
        ])"},
                                            batch2);
    CHECK_STATUS(status);
    std::cout << "===========================" << std::endl;
    std::cout << "     The input batch1 =     " << std::endl;
    std::cout << "===========================" << std::endl;
    batch1->print();
    std::cout << "===========================" << std::endl;
    std::cout << "     The input batch2 =     " << std::endl;
    std::cout << "===========================" << std::endl;
    batch2->print();

    // ===============================================
    //     PUSH THE BATCH TO THE OPERATOR
    // ===============================================

    distinct_operator->add_input(maximus::DeviceTablePtr(std::move(batch1)), 0);

    distinct_operator->add_input(maximus::DeviceTablePtr(std::move(batch2)), 0);

    distinct_operator->no_more_input(0);

    // ===============================================
    //     PRINT THE OUTPUT BATCHES
    // ===============================================
    int num_batches = 0;
    maximus::TableBatchPtr output_batch;
    while (distinct_operator->has_more_batches(true)) {
        output_batch = to_cpu(distinct_operator->export_next_batch(), context, output_schema);
        num_batches++;
    }
    EXPECT_EQ(num_batches, 1);

    // ===============================================
    //     EXPECTED OUTPUT BATCH
    // ===============================================
    maximus::TableBatchPtr expected;
    status = maximus::TableBatch::from_json(context,
                                            output_schema,
                                            {R"([
            ["x", 1, 1],
            ["x", 2, 2],
            ["y", 0, 3],
            ["y", 1, 4],
            ["z", 1, 5],
            ["x", 3, 2],
            ["y", 2, 4]
        ])"},
                                            expected);

    CHECK_STATUS(status);
    std::cout << "===========================" << std::endl;
    std::cout << "     The output batch      " << std::endl;
    std::cout << "===========================" << std::endl;
    output_batch->print();
    std::cout << "===========================" << std::endl;
    std::cout << " The expected output batch " << std::endl;
    std::cout << "===========================" << std::endl;
    expected->print();

    EXPECT_TRUE(*output_batch == *expected);
}

TEST(CudfLimit, MultiTest) {
    // ===============================================
    //     CREATING THE INPUT SCHEMAS
    // ===============================================
    // a source node generating random tuples
    auto fields1 = {
        arrow::field("a", arrow::int8()),  // an integer
        arrow::field("b", arrow::int8()),  // an integer
        arrow::field("c", arrow::int8())   // an integer
    };
    auto input_schema = std::make_shared<maximus::Schema>(fields1);

    // ===============================================
    //     EXPECTED OUTPUT SCHEMA
    // ===============================================
    auto expected_fields = {
        arrow::field("a", arrow::int8()),  // an integer
        arrow::field("b", arrow::int8()),  // an integer
        arrow::field("c", arrow::int8())   // an integer
    };
    auto output_schema = std::make_shared<maximus::Schema>(std::move(expected_fields));

    // ===============================================
    //     CREATING THE LIMIT PROPERTIES
    // ===============================================
    auto limit_properties = std::make_shared<maximus::LimitProperties>(6, 0);

    // std::cout << "Limit properties = \n"
    //           << limit_properties->to_string() << std::endl;

    // ===============================================
    //     CREATING THE OPERATOR
    // ===============================================
    // auto gcontext = maximus::gpu::make_cuda_context();
    auto context = maximus::make_context();
    // context->gcontext = std::move(gcontext);
    auto limit_operator = std::make_shared<maximus::cudf::LimitOperator>(
        context, input_schema, std::move(limit_properties));

    // std::cout << "Finished creating the operator" << std::endl;
    // std::cout << "operator = \n" << limit_operator->to_string() <<
    // std::endl;

    // ===============================================
    //     GENERATE A RANDOM INPUT BATCH
    // ===============================================
    maximus::TableBatchPtr batch1, batch2;
    auto status = maximus::TableBatch::from_json(context,
                                                 input_schema,
                                                 {R"([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1]
        ])"},
                                                 batch1);
    CHECK_STATUS(status);
    std::cout << "===========================" << std::endl;
    std::cout << "    The input batch 1  =   " << std::endl;
    std::cout << "===========================" << std::endl;
    batch1->print();
    status = maximus::TableBatch::from_json(context,
                                            input_schema,
                                            {R"([
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1]
        ])"},
                                            batch2);
    CHECK_STATUS(status);
    std::cout << "===========================" << std::endl;
    std::cout << "   The input batch 2 =    " << std::endl;
    std::cout << "===========================" << std::endl;
    batch2->print();

    // ===============================================
    //     PUSH THE BATCH TO THE OPERATOR
    // ===============================================

    limit_operator->add_input(maximus::DeviceTablePtr(std::move(batch1)), 0);

    limit_operator->add_input(maximus::DeviceTablePtr(std::move(batch2)), 0);

    limit_operator->no_more_input(0);

    // ===============================================
    //     PRINT THE OUTPUT BATCHES
    // ===============================================
    int num_batches = 0;
    maximus::TableBatchPtr output_batch1, output_batch2;
    while (limit_operator->has_more_batches(false)) {
        num_batches == 0
            ? output_batch1 = to_cpu(limit_operator->export_next_batch(), context, output_schema)
            : output_batch2 = to_cpu(limit_operator->export_next_batch(), context, output_schema);
        num_batches++;
    }
    EXPECT_EQ(num_batches, 2);

    // ===============================================
    //     EXPECTED OUTPUT BATCH
    // ===============================================
    maximus::TableBatchPtr expected1, expected2;
    status = maximus::TableBatch::from_json(context,
                                            output_schema,
                                            {R"([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1]
        ])"},
                                            expected1);
    CHECK_STATUS(status);

    status = maximus::TableBatch::from_json(context,
                                            output_schema,
                                            {R"([
            [1, 0, 0],
            [1, 0, 1]
        ])"},
                                            expected2);
    CHECK_STATUS(status);

    std::cout << "===========================" << std::endl;
    std::cout << "     The output batch 1    " << std::endl;
    std::cout << "===========================" << std::endl;
    output_batch1->print();
    std::cout << "===========================" << std::endl;
    std::cout << "     The output batch 2    " << std::endl;
    std::cout << "===========================" << std::endl;
    output_batch2->print();

    std::cout << "===========================" << std::endl;
    std::cout << "The expected output batch 1" << std::endl;
    std::cout << "===========================" << std::endl;
    expected1->print();
    std::cout << "===========================" << std::endl;
    std::cout << "The expected output batch 2" << std::endl;
    std::cout << "===========================" << std::endl;
    expected2->print();

    EXPECT_TRUE(*output_batch1 == *expected1);
    EXPECT_TRUE(*output_batch2 == *expected2);
}

TEST(CudfLocalBroadcast, Test) {
    // ===============================================
    //     CREATING THE INPUT SCHEMAS
    // ===============================================
    // a source node generating random tuples
    auto fields1 = {
        arrow::field("a", arrow::int8()),  // an integer
        arrow::field("b", arrow::int8()),  // an integer
        arrow::field("c", arrow::int8())   // an integer
    };
    auto input_schema = std::make_shared<maximus::Schema>(fields1);

    // ===============================================
    //     EXPECTED OUTPUT SCHEMA
    // ===============================================
    auto expected_fields = {
        arrow::field("a", arrow::int8()),  // an integer
        arrow::field("b", arrow::int8()),  // an integer
        arrow::field("c", arrow::int8())   // an integer
    };
    auto output_schema = std::make_shared<maximus::Schema>(std::move(expected_fields));

    // ===============================================
    //     CREATING THE LIMIT PROPERTIES
    // ===============================================
    auto local_broadcast_properties = std::make_shared<maximus::LocalBroadcastProperties>(3);
    // local_broadcast_properties->should_replicate = true;

    // std::cout << "Local broadcast properties = \n"
    //           << local_broadcast_properties->to_string() << std::endl;

    // ===============================================
    //     CREATING THE OPERATOR
    // ===============================================
    // auto gcontext = maximus::gpu::make_cuda_context();
    auto context = maximus::make_context();
    // context->gcontext = std::move(gcontext);
    auto local_broadcast_operator = std::make_shared<maximus::cudf::LocalBroadcastOperator>(
        context, input_schema, std::move(local_broadcast_properties));

    // std::cout << "Finished creating the operator" << std::endl;
    // std::cout << "operator = \n" << local_broadcast_operator->to_string() <<
    // std::endl;

    // ===============================================
    //     GENERATE A RANDOM INPUT BATCH
    // ===============================================
    maximus::TableBatchPtr batch1, batch2;
    auto status = maximus::TableBatch::from_json(context,
                                                 input_schema,
                                                 {R"([
            [0, 0, 0],
            [0, 0, 1]
        ])"},
                                                 batch1);
    CHECK_STATUS(status);
    std::cout << "===========================" << std::endl;
    std::cout << "    The input batch 1 =    " << std::endl;
    std::cout << "===========================" << std::endl;
    batch1->print();
    status = maximus::TableBatch::from_json(context,
                                            input_schema,
                                            {R"([
            [1, 0, 0],
            [1, 0, 1]
        ])"},
                                            batch2);
    CHECK_STATUS(status);
    std::cout << "===========================" << std::endl;
    std::cout << "   The input batch 2 =    " << std::endl;
    std::cout << "===========================" << std::endl;
    batch2->print();

    // ===============================================
    //     PUSH THE BATCH TO THE OPERATOR
    // ===============================================

    local_broadcast_operator->add_input(maximus::DeviceTablePtr(std::move(batch1)), 0);

    local_broadcast_operator->add_input(maximus::DeviceTablePtr(std::move(batch2)), 0);

    local_broadcast_operator->no_more_input(0);

    // ===============================================
    //     PRINT THE OUTPUT BATCHES
    // ===============================================
    int num_batches = 0;
    maximus::TableBatchPtr output_batch01, output_batch02, output_batch11, output_batch12,
        output_batch21, output_batch22;
    while (local_broadcast_operator->has_more_batches(false, 0)) {
        num_batches == 0
            ? output_batch01 =
                  to_cpu(local_broadcast_operator->export_next_batch(0), context, output_schema)
            : output_batch02 =
                  to_cpu(local_broadcast_operator->export_next_batch(0), context, output_schema);
        num_batches++;
    }
    EXPECT_EQ(num_batches, 2);

    num_batches = 0;
    while (local_broadcast_operator->has_more_batches(false, 1)) {
        num_batches == 0
            ? output_batch11 =
                  to_cpu(local_broadcast_operator->export_next_batch(1), context, output_schema)
            : output_batch12 =
                  to_cpu(local_broadcast_operator->export_next_batch(1), context, output_schema);
        num_batches++;
    }
    EXPECT_EQ(num_batches, 2);

    num_batches = 0;
    while (local_broadcast_operator->has_more_batches(false, 2)) {
        num_batches == 0
            ? output_batch21 =
                  to_cpu(local_broadcast_operator->export_next_batch(2), context, output_schema)
            : output_batch22 =
                  to_cpu(local_broadcast_operator->export_next_batch(2), context, output_schema);
        num_batches++;
    }
    EXPECT_EQ(num_batches, 2);

    // ===============================================
    //     EXPECTED OUTPUT BATCH
    // ===============================================
    maximus::TableBatchPtr expected01, expected02, expected11, expected12, expected21, expected22;
    status = maximus::TableBatch::from_json(context,
                                            output_schema,
                                            {R"([
            [0, 0, 0],
            [0, 0, 1]
        ])"},
                                            expected01);
    CHECK_STATUS(status);

    status = maximus::TableBatch::from_json(context,
                                            output_schema,
                                            {R"([
            [1, 0, 0],
            [1, 0, 1]
        ])"},
                                            expected02);
    CHECK_STATUS(status);

    status = maximus::TableBatch::from_json(context,
                                            output_schema,
                                            {R"([
            [0, 0, 0],
            [0, 0, 1]
        ])"},
                                            expected11);
    CHECK_STATUS(status);

    status = maximus::TableBatch::from_json(context,
                                            output_schema,
                                            {R"([
            [1, 0, 0],
            [1, 0, 1]
        ])"},
                                            expected12);
    CHECK_STATUS(status);

    status = maximus::TableBatch::from_json(context,
                                            output_schema,
                                            {R"([
            [0, 0, 0],
            [0, 0, 1]
        ])"},
                                            expected21);
    CHECK_STATUS(status);

    status = maximus::TableBatch::from_json(context,
                                            output_schema,
                                            {R"([
            [1, 0, 0],
            [1, 0, 1]
        ])"},
                                            expected22);
    CHECK_STATUS(status);

    std::cout << "===========================" << std::endl;
    std::cout << "     The output batch 01   " << std::endl;
    std::cout << "===========================" << std::endl;
    output_batch01->print();
    std::cout << "===========================" << std::endl;
    std::cout << "     The output batch 02   " << std::endl;
    std::cout << "===========================" << std::endl;
    output_batch02->print();
    std::cout << "===========================" << std::endl;
    std::cout << "     The output batch 11   " << std::endl;
    std::cout << "===========================" << std::endl;
    output_batch11->print();
    std::cout << "===========================" << std::endl;
    std::cout << "     The output batch 12   " << std::endl;
    std::cout << "===========================" << std::endl;
    output_batch12->print();
    std::cout << "===========================" << std::endl;
    std::cout << "     The output batch 21   " << std::endl;
    std::cout << "===========================" << std::endl;
    output_batch21->print();
    std::cout << "===========================" << std::endl;
    std::cout << "     The output batch 22   " << std::endl;
    std::cout << "===========================" << std::endl;
    output_batch22->print();

    std::cout << "===========================" << std::endl;
    std::cout << "  The expected output 01   " << std::endl;
    std::cout << "===========================" << std::endl;
    expected01->print();
    std::cout << "===========================" << std::endl;
    std::cout << "  The expected output 02   " << std::endl;
    std::cout << "===========================" << std::endl;
    expected02->print();
    std::cout << "===========================" << std::endl;
    std::cout << "  The expected output 11   " << std::endl;
    std::cout << "===========================" << std::endl;
    expected11->print();
    std::cout << "===========================" << std::endl;
    std::cout << "  The expected output 12   " << std::endl;
    std::cout << "===========================" << std::endl;
    expected12->print();
    std::cout << "===========================" << std::endl;
    std::cout << "  The expected output 21   " << std::endl;
    std::cout << "===========================" << std::endl;
    expected21->print();
    std::cout << "===========================" << std::endl;
    std::cout << "  The expected output 22   " << std::endl;
    std::cout << "===========================" << std::endl;
    expected22->print();

    EXPECT_TRUE(*output_batch01 == *expected01);
    EXPECT_TRUE(*output_batch02 == *expected02);
    EXPECT_TRUE(*output_batch11 == *expected11);
    EXPECT_TRUE(*output_batch12 == *expected12);
    EXPECT_TRUE(*output_batch21 == *expected21);
    EXPECT_TRUE(*output_batch22 == *expected22);
}

}  // namespace test
