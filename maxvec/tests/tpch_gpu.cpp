#include <gtest/gtest.h>

#include <maximus/database.hpp>
#include <maximus/tpch/tpch_queries.hpp>

std::string csv_path() {
    std::string path = PROJECT_SOURCE_DIR;
    path += "/tests/tpch/csv-0.01";
    return path;
}

std::string parquet_path() {
    std::string path = PROJECT_SOURCE_DIR;
    path += "/tests/tpch/parquet";
    return path;
}

TEST(TPCH, Q1) {
    std::string path = csv_path();
    std::cout << "Path = " << path << "\n";

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);

    auto device = maximus::DeviceType::GPU;
    auto q1     = maximus::tpch::q1(db, device);

    std::cout << "Query 1 = \n" << q1->to_string() << std::endl;

    auto table = db->query(q1);

    std::cout << "Query result = \n";
    if (table)
        table->print();
    else
        std::cout << "Query result is empty" << std::endl;

    ASSERT_TRUE(table);
}

TEST(TPCH, Q2) {
    std::string path = csv_path();
    std::cout << "Path = " << path << "\n";

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);

    auto device = maximus::DeviceType::GPU;
    auto q2     = maximus::tpch::q2(db, device);

    std::cout << "Query 2 = \n" << q2->to_string() << std::endl;

    auto table = db->query(q2);

    std::cout << "Query result = \n";
    if (table)
        table->print();
    else
        std::cout << "Query result is empty" << std::endl;

    ASSERT_TRUE(table);
}

TEST(TPCH, Q3) {
    std::string path = csv_path();

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);

    auto device = maximus::DeviceType::GPU;
    auto q3     = maximus::tpch::q3(db, device);

    std::cout << "Query 3 = \n" << q3->to_string() << std::endl;

    auto table = db->query(q3);

    std::cout << "Query result = \n";
    if (table)
        table->print();
    else
        std::cout << "Query result is empty" << std::endl;

    ASSERT_TRUE(table);
}

TEST(TPCH, Q4) {
    std::string path = csv_path();

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);

    auto device = maximus::DeviceType::GPU;
    auto q4     = maximus::tpch::q4(db, device);

    std::cout << "Query 4 = \n" << q4->to_string() << std::endl;

    auto table = db->query(q4);

    std::cout << "Query result = \n";
    if (table)
        table->print();
    else
        std::cout << "Query result is empty" << std::endl;

    ASSERT_TRUE(table);
}

TEST(TPCH, Q5) {
    std::string path = csv_path();

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);

    auto device = maximus::DeviceType::GPU;
    auto q5     = maximus::tpch::q5(db, device);

    std::cout << "Query 5 = \n" << q5->to_string() << std::endl;

    auto table = db->query(q5);

    std::cout << "Query result = \n";
    if (table)
        table->print();
    else
        std::cout << "Query result is empty" << std::endl;

    ASSERT_TRUE(table);
}

TEST(TPCH, Q6) {
    std::string path = csv_path();

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);

    auto device = maximus::DeviceType::GPU;
    auto q6     = maximus::tpch::q6(db, device);

    std::cout << "Query 6 = \n" << q6->to_string() << std::endl;

    auto table = db->query(q6);

    std::cout << "Query result = \n";
    if (table)
        table->print();
    else
        std::cout << "Query result is empty" << std::endl;

    ASSERT_TRUE(table);
}

TEST(TPCH, Q7) {
    std::string path = csv_path();

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);

    auto device = maximus::DeviceType::GPU;
    auto q7     = maximus::tpch::q7(db, device);

    std::cout << "Query 7 = \n" << q7->to_string() << std::endl;

    auto table = db->query(q7);

    std::cout << "Query result = \n";
    if (table)
        table->print();
    else
        std::cout << "Query result is empty" << std::endl;

    ASSERT_TRUE(table);
}

TEST(TPCH, Q8) {
    std::string path = csv_path();

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);

    auto device = maximus::DeviceType::GPU;
    auto q8     = maximus::tpch::q8(db, device);

    std::cout << "Query 8 = \n" << q8->to_string() << std::endl;

    auto table = db->query(q8);

    std::cout << "Query result = \n";
    if (table)
        table->print();
    else
        std::cout << "Query result is empty" << std::endl;

    ASSERT_TRUE(table);
}

TEST(TPCH, Q9) {
    std::string path = csv_path();

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);

    auto device = maximus::DeviceType::GPU;
    auto q9     = maximus::tpch::q9(db, device);

    std::cout << "Query 9 = \n" << q9->to_string() << std::endl;

    auto table = db->query(q9);

    std::cout << "Query result = \n";
    if (table)
        table->print();
    else
        std::cout << "Query result is empty" << std::endl;

    ASSERT_TRUE(table);
}

TEST(TPCH, Q10) {
    std::string path = csv_path();

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);

    auto device = maximus::DeviceType::GPU;
    auto q10    = maximus::tpch::q10(db, device);

    std::cout << "Query 10 = \n" << q10->to_string() << std::endl;

    auto table = db->query(q10);

    std::cout << "Query result = \n";
    if (table)
        table->print();
    else
        std::cout << "Query result is empty" << std::endl;

    ASSERT_TRUE(table);
}

TEST(TPCH, Q11) {
    std::string path = csv_path();

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);

    auto device = maximus::DeviceType::GPU;
    auto q11    = maximus::tpch::q11(db, device);

    std::cout << "Query 11 = \n" << q11->to_string() << std::endl;

    auto table = db->query(q11);

    std::cout << "Query result = \n";
    if (table)
        table->print();
    else
        std::cout << "Query result is empty" << std::endl;

    ASSERT_TRUE(table);
}

TEST(TPCH, Q12) {
    std::string path = csv_path();

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);

    auto device = maximus::DeviceType::GPU;
    auto q12    = maximus::tpch::q12(db, device);

    std::cout << "Query 12 = \n" << q12->to_string() << std::endl;

    auto table = db->query(q12);

    std::cout << "Query result = \n";
    if (table)
        table->print();
    else
        std::cout << "Query result is empty" << std::endl;

    ASSERT_TRUE(table);
}

TEST(TPCH, Q13) {
    std::string path = csv_path();

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);

    auto device = maximus::DeviceType::GPU;
    auto q13    = maximus::tpch::q13(db, device);

    std::cout << "Query 13 = \n" << q13->to_string() << std::endl;

    auto table = db->query(q13);

    std::cout << "Query result = \n";
    if (table)
        table->print();
    else
        std::cout << "Query result is empty" << std::endl;

    ASSERT_TRUE(table);
}

TEST(TPCH, Q14) {
    std::string path = csv_path();

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);

    auto device = maximus::DeviceType::GPU;
    auto q14    = maximus::tpch::q14(db, device);

    std::cout << "Query 14 = \n" << q14->to_string() << std::endl;

    auto table = db->query(q14);

    std::cout << "Query result = \n";
    if (table)
        table->print();
    else
        std::cout << "Query result is empty" << std::endl;

    ASSERT_TRUE(table);
}

TEST(TPCH, Q15) {
    std::string path = csv_path();
    std::cout << "Path = " << path << "\n";

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);

    auto device = maximus::DeviceType::GPU;
    auto q15    = maximus::tpch::q15(db, device);

    std::cout << "Query 15 = \n" << q15->to_string() << std::endl;

    auto table = db->query(q15);

    std::cout << "Query result = \n";
    if (table)
        table->print();
    else {
        std::cout << "Query result is empty" << std::endl;
        std::cout << "Output schema = " << q15->get_input_schemas()[0]->to_string() << std::endl;
    }

    // this query is empty because the testing data set it too small
    ASSERT_TRUE(table);
}

TEST(TPCH, Q16) {
    std::string path = csv_path();
    std::cout << "Path = " << path << "\n";

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);

    auto device = maximus::DeviceType::GPU;
    auto q16    = maximus::tpch::q16(db, device);

    std::cout << "Query 16 = \n" << q16->to_string() << std::endl;

    auto table = db->query(q16);

    std::cout << "Query result = \n";
    if (table)
        table->print();
    else {
        std::cout << "Query result is empty" << std::endl;
        std::cout << "Output schema = " << q16->get_input_schemas()[0]->to_string() << std::endl;
    }

    // this query is empty because the testing data set it too small
    ASSERT_TRUE(table);
}

TEST(TPCH, Q17) {
    std::string path = csv_path();
    std::cout << "Path = " << path << "\n";

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);

    auto device = maximus::DeviceType::GPU;
    auto q17    = maximus::tpch::q17(db, device);

    std::cout << "Query 17 = \n" << q17->to_string() << std::endl;

    auto table = db->query(q17);

    std::cout << "Query result = \n";
    if (table)
        table->print();
    else {
        std::cout << "Query result is empty" << std::endl;
        std::cout << "Output schema = " << q17->get_input_schemas()[0]->to_string() << std::endl;
    }

    // this query is empty because the testing data set it too small
    ASSERT_FALSE(table);
}

TEST(TPCH, Q18) {
    std::string path = csv_path();

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);

    auto device = maximus::DeviceType::GPU;
    auto q18    = maximus::tpch::q18(db, device);

    std::cout << "Query 18 = \n" << q18->to_string() << std::endl;

    auto table = db->query(q18);

    std::cout << "Query result = \n";
    if (table)
        table->print();
    else
        std::cout << "Query result is empty" << std::endl;

    ASSERT_TRUE(table);
}

TEST(TPCH, Q19) {
    std::string path = csv_path();

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);

    auto device = maximus::DeviceType::GPU;
    auto q19    = maximus::tpch::q19(db, device);

    std::cout << "Query 19 = \n" << q19->to_string() << std::endl;

    auto table = db->query(q19);

    std::cout << "Query result = \n";
    if (table)
        table->print();
    else
        std::cout << "Query result is empty" << std::endl;

    ASSERT_TRUE(table);
}

TEST(TPCH, Q20) {
    std::string path  = csv_path();
    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);

    auto device = maximus::DeviceType::GPU;
    auto q20    = maximus::tpch::q20(db, device);

    std::cout << "Query 20 = \n" << q20->to_string() << std::endl;

    auto table = db->query(q20);

    std::cout << "Query result = \n";
    if (table)
        table->print();
    else
        std::cout << "Query result is empty" << std::endl;

    ASSERT_TRUE(table);
}

TEST(TPCH, Q21) {
    std::string path  = csv_path();
    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);

    auto device = maximus::DeviceType::GPU;
    auto q21    = maximus::tpch::q21(db, device);

    std::cout << "Query 21 = \n" << q21->to_string() << std::endl;

    auto table = db->query(q21);

    std::cout << "Query result = \n";
    if (table)
        table->print();
    else
        std::cout << "Query result is empty" << std::endl;

    ASSERT_TRUE(table);
}

TEST(TPCH, Q22) {
    std::string path  = csv_path();
    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);

    auto device = maximus::DeviceType::GPU;
    auto q22    = maximus::tpch::q22(db, device);

    std::cout << "Query 22 = \n" << q22->to_string() << std::endl;

    auto table = db->query(q22);

    std::cout << "Query result = \n";
    if (table)
        table->print();
    else
        std::cout << "Query result is empty" << std::endl;

    ASSERT_TRUE(table);
}
