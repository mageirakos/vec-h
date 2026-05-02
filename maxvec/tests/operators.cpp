#include <gtest/gtest.h>

#include <maximus/context.hpp>
#include <maximus/dag/query_plan.hpp>
#include <maximus/operators/acero/group_by_operator.hpp>
#include <maximus/operators/native/scatter_operator.hpp>
#include <maximus/operators/native/gather_operator.hpp>
#include <maximus/types/aggregate.hpp>
#include <maximus/types/table_batch.hpp>

#ifdef MAXIMUS_WITH_CUDA
#include <maximus/gpu/cuda_api.hpp>
#include <maximus/operators/gpu/cudf/scatter_operator.hpp>
#include <maximus/operators/gpu/cudf/gather_operator.hpp>
#endif

namespace test {

TEST(Operators, AceroGroupBy) {
    // ===============================================
    //     CREATING THE INPUT SCHEMA
    // ===============================================
    // a source node generating random tuples
    auto fields = {
        arrow::field("key", arrow::utf8()),    //  a string
        arrow::field("value", arrow::int32())  // an integer
    };
    auto input_schema = std::make_shared<maximus::Schema>(fields);

    // ===============================================
    //     CREATING THE AGGREGATES
    // ===============================================
    auto arrow_aggregate =
        std::make_shared<arrow::compute::Aggregate>("hash_sum", "value", "value_sum");

    auto aggregate = std::make_shared<maximus::Aggregate>(std::move(arrow_aggregate));

    // ===============================================
    //     CREATING THE GROUP KEYS
    // ===============================================
    std::vector<arrow::FieldRef> group_keys                     = {"key"};
    std::vector<std::shared_ptr<maximus::Aggregate>> aggregates = {std::move(aggregate)};
    // std::cout << "Group keys[0] = " << group_keys[0] << std::endl;

    // ===============================================
    //     WRAPPING INSIDE THE GROUP BY PROPERTIES
    // ===============================================
    auto group_by_properties =
        std::make_shared<maximus::GroupByProperties>(std::move(group_keys), std::move(aggregates));
    // std::cout << "Group by properties = \n" << group_by_properties->to_string() << std::endl;

    // ===============================================
    //     CREATING THE OPERATOR
    // ===============================================
    auto context           = maximus::make_context();
    auto group_by_operator = std::make_shared<maximus::acero::GroupByOperator>(
        context, input_schema, std::move(group_by_properties));

    group_by_operator->next_op_type     = maximus::PhysicalOperatorType::TABLE_SINK;
    group_by_operator->next_engine_type = maximus::EngineType::NATIVE;

    std::cout << "Finished creating the operator" << std::endl;
    std::cout << "operator = \n" << group_by_operator->to_string() << std::endl;

    // ===============================================
    //     GENERATE A RANDOM INPUT BATCH
    // ===============================================
    maximus::TableBatchPtr batch;
    auto status = maximus::TableBatch::from_json(context,
                                                 input_schema,
                                                 {R"([
            ["x", 1],
            ["y", 2],
            ["y", 3],
            ["z", 4],
            ["z", 5]
        ])"},
                                                 batch);
    CHECK_STATUS(status);
    std::cout << "===========================" << std::endl;
    std::cout << "     The input batch =     " << std::endl;
    std::cout << "===========================" << std::endl;
    batch->print();

    // ===============================================
    //     PUSH THE BATCH TO THE OPERATOR
    // ===============================================
    group_by_operator->add_input(maximus::DeviceTablePtr(std::move(batch)), 0);
    group_by_operator->no_more_input(0);

    // ===============================================
    //     PRINT THE OUTPUT BATCHES
    // ===============================================
    int num_batches = 0;
    maximus::DeviceTablePtr output_batch;
    while (group_by_operator->has_more_batches(true)) {
        output_batch = group_by_operator->export_next_batch();
        num_batches++;
    }
    EXPECT_EQ(num_batches, 1);

    // ===============================================
    //     EXPECTED OUTPUT SCHEMA
    // ===============================================
    // note that the value_sum column has a different name than the input schema
    auto expected_fields = {
        arrow::field("key", arrow::utf8()),        //  a string
        arrow::field("value_sum", arrow::int32())  // an integer
    };
    auto output_schema = std::make_shared<maximus::Schema>(expected_fields);

    // ===============================================
    //     EXPECTED OUTPUT BATCH
    // ===============================================
    maximus::TableBatchPtr expected;
    status = maximus::TableBatch::from_json(context,
                                            output_schema,
                                            {R"([
            ["x", 1],
            ["y", 5],
            ["z", 9]
        ])"},
                                            expected);

    CHECK_STATUS(status);
    std::cout << "===========================" << std::endl;
    std::cout << "     The output batch      " << std::endl;
    std::cout << "===========================" << std::endl;
    output_batch.to_cpu(context, output_schema);
    output_batch.as_cpu()->print();
    std::cout << "===========================" << std::endl;
    std::cout << " The expected output batch " << std::endl;
    std::cout << "===========================" << std::endl;
    expected->print();

    EXPECT_TRUE(*output_batch.as_cpu() == *expected);
}


TEST(Operators, NativeScatterAndGatherMissingPartition) {
    // ===============================================
    //     CREATING THE INPUT SCHEMA
    // ===============================================
    auto fields = {
        arrow::field("query_id", arrow::int32()),
        arrow::field("value", arrow::int32())
    };
    auto input_schema = std::make_shared<maximus::Schema>(fields);

    // ===============================================
    //     CREATING THE PROPERTIES (3 unique keys -> 3 partitions)
    // ===============================================
    int num_partitions = 3;
    auto partition_properties = std::make_shared<maximus::ScatterProperties>(
        std::vector<std::string>{"query_id"}, num_partitions);

    // ===============================================
    //     CREATING THE OPERATOR
    // ===============================================
    auto context = maximus::make_context();
    auto partition_op = std::make_shared<maximus::native::ScatterOperator>(
        context, input_schema, std::move(partition_properties));

    partition_op->next_op_type = maximus::PhysicalOperatorType::TABLE_SINK;
    partition_op->next_engine_type = maximus::EngineType::NATIVE;

    std::cout << "Finished creating the operator" << std::endl;

    // ===============================================
    //     GENERATE INPUT BATCH (unsorted by query_id)
    // ===============================================
    maximus::TableBatchPtr batch;
    auto status = maximus::TableBatch::from_json(context,
                                                 input_schema,
                                                 {R"([
            [1, 200],
            [3, 300],
            [1, 400],
            [3, 600]
        ])"},
                                                 batch);
    CHECK_STATUS(status);
    std::cout << "===========================\n";
    std::cout << "     The input batch =     \n";
    std::cout << "===========================\n";
    batch->print();

    // ===============================================
    //     PUSH THE BATCH TO THE OPERATOR
    // ===============================================
    partition_op->add_input(maximus::DeviceTablePtr(std::move(batch)), 0);
    partition_op->no_more_input(0);

    // ===============================================
    //     GET OUTPUT FROM EACH PORT (multi-port)
    // ===============================================
    
    std::cout << "===========================\n";
    std::cout << " Partition outputs:        \n";
    std::cout << "===========================\n";


    // Port 0: query_id=1 rows
    EXPECT_TRUE(partition_op->has_more_batches(true, 0));
    auto port0_batch = partition_op->export_next_batch(0);
    port0_batch.to_cpu(context, input_schema);
    std::cout << "Port 0 (query_id=1):\n";
    port0_batch.as_cpu()->print();
    EXPECT_EQ(port0_batch.as_cpu()->num_rows(), 2);

    // Port 1: query_id=2 rows
    EXPECT_TRUE(partition_op->has_more_batches(true, 1));
    auto port1_batch = partition_op->export_next_batch(1);
    port1_batch.to_cpu(context, input_schema);
    std::cout << "Port 1 (query_id=3):\n";
    port1_batch.as_cpu()->print();
    EXPECT_EQ(port1_batch.as_cpu()->num_rows(), 2);

    // Port 2: query_id=3 rows
    // EXPECT_FALSE(partition_op->has_more_batches(false, 2));
    // auto port2_batch = partition_op->export_next_batch(2);
    // port2_batch.to_cpu(context, input_schema);
    // std::cout << "Port 2 (query_id=N/A):\n";
    // port2_batch.as_cpu()->print();
    // EXPECT_EQ(port2_batch.as_cpu()->num_rows(), 0);

    // ===============================================
    //     Gather back and verify
    // ===============================================
    auto union_properties = std::make_shared<maximus::GatherProperties>(num_partitions);
    auto union_op = std::make_shared<maximus::native::GatherOperator>(
        context, input_schema, std::move(union_properties));
    union_op->next_op_type = maximus::PhysicalOperatorType::TABLE_SINK;
    union_op->next_engine_type = maximus::EngineType::NATIVE;

    union_op->add_input(port0_batch, 0);
    union_op->add_input(port1_batch, 1);
    // union_op->add_input(port2_batch, 2);
    union_op->no_more_input(0);
    union_op->no_more_input(1);
    union_op->no_more_input(2); // <-- still need to ensure port is closed

    auto output_batch = union_op->export_next_batch();
    output_batch.to_cpu(context, input_schema);
    
    std::cout << "===========================\n";
    std::cout << "     The output batch      \n";
    std::cout << "===========================\n";
    output_batch.as_cpu()->print();

    // ===============================================
    //     EXPECTED OUTPUT (sorted by query_id after scatter+gather)
    // ===============================================
    maximus::TableBatchPtr expected;
    status = maximus::TableBatch::from_json(context,
                                            input_schema,
                                            {R"([
            [1, 200],
            [1, 400],
            [3, 300],
            [3, 600]
        ])"},
                                            expected);

    CHECK_STATUS(status);
    std::cout << "===========================\n";
    std::cout << " The expected output batch \n";
    std::cout << "===========================\n";
    expected->print();

    EXPECT_TRUE(*output_batch.as_cpu() == *expected);
}



TEST(Operators, NativeScatterAndGather) {
    // ===============================================
    //     CREATING THE INPUT SCHEMA
    // ===============================================
    auto fields = {
        arrow::field("query_id", arrow::int32()),
        arrow::field("value", arrow::int32())
    };
    auto input_schema = std::make_shared<maximus::Schema>(fields);

    // ===============================================
    //     CREATING THE PROPERTIES (3 unique keys -> 3 partitions)
    // ===============================================
    int num_partitions = 3;
    auto partition_properties = std::make_shared<maximus::ScatterProperties>(
        std::vector<std::string>{"query_id"}, num_partitions);

    // ===============================================
    //     CREATING THE OPERATOR
    // ===============================================
    auto context = maximus::make_context();
    auto partition_op = std::make_shared<maximus::native::ScatterOperator>(
        context, input_schema, std::move(partition_properties));

    partition_op->next_op_type = maximus::PhysicalOperatorType::TABLE_SINK;
    partition_op->next_engine_type = maximus::EngineType::NATIVE;

    std::cout << "Finished creating the operator" << std::endl;

    // ===============================================
    //     GENERATE INPUT BATCH (unsorted by query_id)
    // ===============================================
    maximus::TableBatchPtr batch;
    auto status = maximus::TableBatch::from_json(context,
                                                 input_schema,
                                                 {R"([
            [9, 100],
            [1, 200],
            [3, 300],
            [1, 400],
            [9, 500],
            [3, 600]
        ])"},
                                                 batch);
    CHECK_STATUS(status);
    std::cout << "===========================\n";
    std::cout << "     The input batch =     \n";
    std::cout << "===========================\n";
    batch->print();

    // ===============================================
    //     PUSH THE BATCH TO THE OPERATOR
    // ===============================================
    partition_op->add_input(maximus::DeviceTablePtr(std::move(batch)), 0);
    partition_op->no_more_input(0);

    // ===============================================
    //     GET OUTPUT FROM EACH PORT (multi-port)
    // ===============================================
    
    std::cout << "===========================\n";
    std::cout << " Partition outputs:        \n";
    std::cout << "===========================\n";


    // Port 0: query_id=1 rows
    EXPECT_TRUE(partition_op->has_more_batches(true, 0));
    auto port0_batch = partition_op->export_next_batch(0);
    port0_batch.to_cpu(context, input_schema);
    std::cout << "Port 0 (query_id=1):\n";
    port0_batch.as_cpu()->print();
    EXPECT_EQ(port0_batch.as_cpu()->num_rows(), 2);

    // Port 1: query_id=2 rows
    EXPECT_TRUE(partition_op->has_more_batches(true, 1));
    auto port1_batch = partition_op->export_next_batch(1);
    port1_batch.to_cpu(context, input_schema);
    std::cout << "Port 1 (query_id=3):\n";
    port1_batch.as_cpu()->print();
    EXPECT_EQ(port1_batch.as_cpu()->num_rows(), 2);

    // Port 2: query_id=2 rows
    EXPECT_TRUE(partition_op->has_more_batches(true, 2));
    auto port2_batch = partition_op->export_next_batch(2);
    port2_batch.to_cpu(context, input_schema);
    std::cout << "Port 2 (query_id=9):\n";
    port2_batch.as_cpu()->print();
    EXPECT_EQ(port2_batch.as_cpu()->num_rows(), 2);

    // ===============================================
    //     Gather back and verify
    // ===============================================
    auto union_properties = std::make_shared<maximus::GatherProperties>(num_partitions);
    auto union_op = std::make_shared<maximus::native::GatherOperator>(
        context, input_schema, std::move(union_properties));
    union_op->next_op_type = maximus::PhysicalOperatorType::TABLE_SINK;
    union_op->next_engine_type = maximus::EngineType::NATIVE;

    union_op->add_input(port0_batch, 0);
    union_op->add_input(port1_batch, 1);
    union_op->add_input(port2_batch, 2);
    union_op->no_more_input(0);
    union_op->no_more_input(1);
    union_op->no_more_input(2);

    auto output_batch = union_op->export_next_batch();
    output_batch.to_cpu(context, input_schema);
    
    std::cout << "===========================\n";
    std::cout << "     The output batch      \n";
    std::cout << "===========================\n";
    output_batch.as_cpu()->print();

    // ===============================================
    //     EXPECTED OUTPUT (sorted by query_id after scatter+gather)
    // ===============================================
    maximus::TableBatchPtr expected;
    status = maximus::TableBatch::from_json(context,
                                            input_schema,
                                            {R"([
            [1, 200],
            [1, 400],
            [3, 300],
            [3, 600],
            [9, 100],
            [9, 500]
        ])"},
                                            expected);

    CHECK_STATUS(status);
    std::cout << "===========================\n";
    std::cout << " The expected output batch \n";
    std::cout << "===========================\n";
    expected->print();

    EXPECT_TRUE(*output_batch.as_cpu() == *expected);
}

TEST(Operators, NativeGather) {
    // ===============================================
    // Test Gather operator - collects from multiple ports
    // ===============================================
    auto fields = {
        arrow::field("query_id", arrow::int64()),
        arrow::field("value", arrow::int32())
    };
    auto input_schema = std::make_shared<maximus::Schema>(fields);

    int num_inputs = 3;
    auto union_properties = std::make_shared<maximus::GatherProperties>(num_inputs);

    auto context = maximus::make_context();
    auto union_op = std::make_shared<maximus::native::GatherOperator>(
        context, input_schema, std::move(union_properties));

    union_op->next_op_type = maximus::PhysicalOperatorType::TABLE_SINK;
    union_op->next_engine_type = maximus::EngineType::NATIVE;

    std::cout << "Testing Gather with 3 input ports" << std::endl;

    // Create input batches for each port
    maximus::TableBatchPtr batch0, batch1, batch2;
    
    auto status = maximus::TableBatch::from_json(context, input_schema,
        {R"([[0, 100], [0, 200]])"}, batch0);
    CHECK_STATUS(status);
    
    status = maximus::TableBatch::from_json(context, input_schema,
        {R"([[1, 300], [1, 400]])"}, batch1);
    CHECK_STATUS(status);
    
    status = maximus::TableBatch::from_json(context, input_schema,
        {R"([[2, 500]])"}, batch2);
    CHECK_STATUS(status);

    std::cout << "===========================\n";
    std::cout << "     Input batches:        \n";
    std::cout << "===========================\n";
    std::cout << "Port 0:\n"; batch0->print();
    std::cout << "Port 1:\n"; batch1->print();
    std::cout << "Port 2:\n"; batch2->print();

    // Add inputs from all ports
    union_op->add_input(maximus::DeviceTablePtr(std::move(batch0)), 0);
    union_op->add_input(maximus::DeviceTablePtr(std::move(batch1)), 1);
    union_op->add_input(maximus::DeviceTablePtr(std::move(batch2)), 2);
    
    // Signal no more input for all ports
    union_op->no_more_input(0);
    union_op->no_more_input(1);
    union_op->no_more_input(2);

    // Get concatenated output
    EXPECT_TRUE(union_op->has_more_batches(true));
    auto output_batch = union_op->export_next_batch();
    output_batch.to_cpu(context, input_schema);
    
    std::cout << "===========================\n";
    std::cout << "     Concatenated output:  \n";
    std::cout << "===========================\n";
    output_batch.as_cpu()->print();
    
    EXPECT_EQ(output_batch.as_cpu()->num_rows(), 5);  // 2 + 2 + 1 = 5 rows

    // ===============================================
    // Verify exact expected output
    // ===============================================
    maximus::TableBatchPtr expected;
    status = maximus::TableBatch::from_json(context,
                                            input_schema,
                                            {R"([
            [0, 100],
            [0, 200],
            [1, 300],
            [1, 400],
            [2, 500]
        ])"},
                                            expected);
    CHECK_STATUS(status);
    std::cout << "===========================\n";
    std::cout << " Expected output:          \n";
    std::cout << "===========================\n";
    expected->print();

    EXPECT_TRUE(*output_batch.as_cpu() == *expected);
}

#ifdef MAXIMUS_WITH_CUDA
TEST(Operators, CudfScatterKeyAgnostic) {
    // ===============================================
    // Test GPU key-agnostic partitioning with non-sequential keys
    // Keys: [50, 100, 150] should map to ports [0, 1, 2]
    // ===============================================
    auto fields = {
        arrow::field("query_id", arrow::int64()),
        arrow::field("value", arrow::int32())
    };
    auto input_schema = std::make_shared<maximus::Schema>(fields);

    // 3 partitions for keys 50, 100, 150
    int num_partitions = 3;
    auto partition_properties = std::make_shared<maximus::ScatterProperties>(
        std::vector<std::string>{"query_id"}, num_partitions);

    auto context = maximus::make_context();
    
    auto partition_op = std::make_shared<maximus::cudf::ScatterOperator>(
        context, input_schema, std::move(partition_properties));

    partition_op->next_op_type = maximus::PhysicalOperatorType::TABLE_SINK;
    partition_op->next_engine_type = maximus::EngineType::CUDF;

    std::cout << "Testing GPU key-agnostic partitioning with keys [50, 100, 150]" << std::endl;

    // Input with non-sequential keys
    maximus::TableBatchPtr batch;
    auto status = maximus::TableBatch::from_json(context,
                                                 input_schema,
                                                 {R"([
            [100, 200],
            [50, 100],
            [150, 300],
            [50, 400],
            [100, 500],
            [150, 600],
            [50, 700]
        ])"},
                                                 batch);
    CHECK_STATUS(status);
    std::cout << "===========================\n";
    std::cout << "     The input batch =     \n";
    std::cout << "===========================\n";
    batch->print();

    partition_op->add_input(maximus::DeviceTablePtr(std::move(batch)), 0);
    partition_op->no_more_input(0);

    std::cout << "===========================\n";
    std::cout << " GPU Partition outputs:    \n";
    std::cout << "===========================\n";

    // Port 0 should have query_id=50 rows (sorted keys: 50 < 100 < 150)
    EXPECT_TRUE(partition_op->has_more_batches(true, 0));
    auto port0_batch = partition_op->export_next_batch(0);
    port0_batch.to_cpu(context, input_schema);
    std::cout << "Port 0 (query_id=50, smallest key):\n";
    port0_batch.as_cpu()->print();
    EXPECT_EQ(port0_batch.as_cpu()->num_rows(), 3);  // 3 rows with key=50

    // Port 1 should have query_id=100 rows
    EXPECT_TRUE(partition_op->has_more_batches(true, 1));
    auto port1_batch = partition_op->export_next_batch(1);
    port1_batch.to_cpu(context, input_schema);
    std::cout << "Port 1 (query_id=100, middle key):\n";
    port1_batch.as_cpu()->print();
    EXPECT_EQ(port1_batch.as_cpu()->num_rows(), 2);  // 2 rows with key=100

    // Port 2 should have query_id=150 rows
    EXPECT_TRUE(partition_op->has_more_batches(true, 2));
    auto port2_batch = partition_op->export_next_batch(2);
    port2_batch.to_cpu(context, input_schema);
    std::cout << "Port 2 (query_id=150, largest key):\n";
    port2_batch.as_cpu()->print();
    EXPECT_EQ(port2_batch.as_cpu()->num_rows(), 2);  // 2 rows with key=150

    // ===============================================
    // Now gather all partitions back together on GPU
    // ===============================================
    std::cout << "===========================\n";
    std::cout << " GPU Gather all partitions \n";
    std::cout << "===========================\n";

    auto union_properties = std::make_shared<maximus::GatherProperties>(num_partitions);
    auto union_op = std::make_shared<maximus::cudf::GatherOperator>(
        context, input_schema, std::move(union_properties));
    union_op->next_op_type = maximus::PhysicalOperatorType::TABLE_SINK;
    union_op->next_engine_type = maximus::EngineType::CUDF;

    // Feed partitioned outputs into union
    union_op->add_input(port0_batch, 0);
    union_op->add_input(port1_batch, 1);
    union_op->add_input(port2_batch, 2);
    union_op->no_more_input(0);
    union_op->no_more_input(1);
    union_op->no_more_input(2);

    EXPECT_TRUE(union_op->has_more_batches(true));
    auto final_batch = union_op->export_next_batch();
    final_batch.to_cpu(context, input_schema);
    std::cout << "GPU Final merged output:\n";
    final_batch.as_cpu()->print();

    // Should have all 7 original rows
    EXPECT_EQ(final_batch.as_cpu()->num_rows(), 7);

    // ===============================================
    // Verify exact expected output
    // After scatter + gather: rows grouped by key in sorted order
    // ===============================================
    maximus::TableBatchPtr expected;
    status = maximus::TableBatch::from_json(context,
                                            input_schema,
                                            {R"([
            [50, 100],
            [50, 400],
            [50, 700],
            [100, 200],
            [100, 500],
            [150, 300],
            [150, 600]
        ])"},
                                            expected);
    CHECK_STATUS(status);
    std::cout << "===========================\n";
    std::cout << " Expected merged output:   \n";
    std::cout << "===========================\n";
    expected->print();

    EXPECT_TRUE(*final_batch.as_cpu() == *expected);
}

TEST(Operators, CudfGather) {
    // ===============================================
    // Test GPU Gather operator - collects from multiple ports
    // ===============================================
    auto fields = {
        arrow::field("query_id", arrow::int64()),
        arrow::field("value", arrow::int32())
    };
    auto input_schema = std::make_shared<maximus::Schema>(fields);

    int num_inputs = 3;
    auto union_properties = std::make_shared<maximus::GatherProperties>(num_inputs);

    auto context = maximus::make_context();
    
    auto union_op = std::make_shared<maximus::cudf::GatherOperator>(
        context, input_schema, std::move(union_properties));

    union_op->next_op_type = maximus::PhysicalOperatorType::TABLE_SINK;
    union_op->next_engine_type = maximus::EngineType::CUDF;

    std::cout << "Testing GPU Gather with 3 input ports" << std::endl;

    // Create input batches for each port
    maximus::TableBatchPtr batch0, batch1, batch2;
    
    auto status = maximus::TableBatch::from_json(context, input_schema,
        {R"([[0, 100], [0, 200]])"}, batch0);
    CHECK_STATUS(status);
    
    status = maximus::TableBatch::from_json(context, input_schema,
        {R"([[1, 300], [1, 400]])"}, batch1);
    CHECK_STATUS(status);
    
    status = maximus::TableBatch::from_json(context, input_schema,
        {R"([[2, 500]])"}, batch2);
    CHECK_STATUS(status);

    std::cout << "===========================\n";
    std::cout << "     Input batches:        \n";
    std::cout << "===========================\n";
    std::cout << "Port 0:\n"; batch0->print();
    std::cout << "Port 1:\n"; batch1->print();
    std::cout << "Port 2:\n"; batch2->print();

    // Add inputs from all ports
    union_op->add_input(maximus::DeviceTablePtr(std::move(batch0)), 0);
    union_op->add_input(maximus::DeviceTablePtr(std::move(batch1)), 1);
    union_op->add_input(maximus::DeviceTablePtr(std::move(batch2)), 2);
    
    // Signal no more input for all ports
    union_op->no_more_input(0);
    union_op->no_more_input(1);
    union_op->no_more_input(2);

    // Get concatenated output
    EXPECT_TRUE(union_op->has_more_batches(true));
    auto output_batch = union_op->export_next_batch();
    output_batch.to_cpu(context, input_schema);
    
    std::cout << "===========================\n";
    std::cout << " GPU Concatenated output:  \n";
    std::cout << "===========================\n";
    output_batch.as_cpu()->print();
    
    EXPECT_EQ(output_batch.as_cpu()->num_rows(), 5);  // 2 + 2 + 1 = 5 rows

    // ===============================================
    // Verify exact expected output
    // ===============================================
    maximus::TableBatchPtr expected;
    status = maximus::TableBatch::from_json(context,
                                            input_schema,
                                            {R"([
            [0, 100],
            [0, 200],
            [1, 300],
            [1, 400],
            [2, 500]
        ])"},
                                            expected);
    CHECK_STATUS(status);
    std::cout << "===========================\n";
    std::cout << " Expected output:          \n";
    std::cout << "===========================\n";
    expected->print();

    EXPECT_TRUE(*output_batch.as_cpu() == *expected);
}
#endif

}  // namespace test
