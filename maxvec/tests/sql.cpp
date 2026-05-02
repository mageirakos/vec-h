#include <arrow/acero/hash_join.h>
#include <arrow/acero/util.h>
#include <arrow/result.h>
#include <gtest/gtest.h>

#include <maximus/database.hpp>
#include <maximus/database_catalogue.hpp>

using namespace maximus;

namespace test {

void set_tpch_schemas(std::shared_ptr<maximus::Database> db) {
    // todo: maybe some types should be changed but
    // this is working for now, and we can leave it for future
    db->parse_schema("CREATE TABLE nation ( "
                     "n_nationkey  INTEGER       NOT NULL, "
                     "n_name       CHAR(25)      NOT NULL, "
                     "n_regionkey  INTEGER       NOT NULL, "
                     "n_comment    VARCHAR(152), "
                     "PRIMARY KEY (n_nationkey) "
                     "); ");

    db->parse_schema("CREATE TABLE region ( "
                     "r_regionkey  INTEGER       NOT NULL, "
                     "r_name       CHAR(25)      NOT NULL, "
                     "r_comment    VARCHAR(152), "
                     "PRIMARY KEY (r_regionkey) "
                     "); ");

    db->parse_schema("CREATE TABLE part ( "
                     "p_partkey     INTEGER       NOT NULL, "
                     "p_name        VARCHAR(55)   NOT NULL, "
                     "p_mfgr        CHAR(25)      NOT NULL, "
                     "p_brand       CHAR(10)      NOT NULL, "
                     "p_type        VARCHAR(25)   NOT NULL, "
                     "p_size        INTEGER       NOT NULL, "
                     "p_container   CHAR(10)      NOT NULL, "
                     // "p_retailprice DECIMAL(15,2) NOT NULL, "
                     "p_retailprice DOUBLE        NOT NULL, "
                     "p_comment     VARCHAR(23)   NOT NULL, "
                     "PRIMARY KEY (p_partkey) "
                     "); ");

    db->parse_schema("CREATE TABLE supplier ( "
                     "s_suppkey   INTEGER       NOT NULL, "
                     "s_name      CHAR(25)      NOT NULL, "
                     "s_address   VARCHAR(40)   NOT NULL, "
                     "s_nationkey INTEGER       NOT NULL, "
                     "s_phone     CHAR(15)      NOT NULL, "
                     // "s_acctbal   DECIMAL(15,2) NOT NULL, "
                     "s_acctbal   DOUBLE        NOT NULL, "
                     "s_comment   VARCHAR(101)  NOT NULL, "
                     "PRIMARY KEY (s_suppkey) "
                     "); ");

    db->parse_schema("CREATE TABLE partsupp ( "
                     "ps_partkey    INTEGER       NOT NULL, "
                     "ps_suppkey    INTEGER       NOT NULL, "
                     // "ps_availqty   INTEGER       NOT NULL, "
                     "ps_availqty   DOUBLE        NOT NULL, "
                     // "ps_supplycost DECIMAL(15,2) NOT NULL, "
                     "ps_supplycost DOUBLE        NOT NULL, "
                     "ps_comment    VARCHAR(199)  NOT NULL, "
                     "PRIMARY KEY (ps_partkey, ps_suppkey) "
                     "); ");

    db->parse_schema("CREATE TABLE customer ( "
                     "c_custkey    INTEGER       NOT NULL, "
                     "c_name       VARCHAR(25)   NOT NULL, "
                     "c_address    VARCHAR(40)   NOT NULL, "
                     "c_nationkey  INTEGER       NOT NULL, "
                     "c_phone      CHAR(15)      NOT NULL, "
                     // "c_acctbal    DECIMAL(15,2) NOT NULL, "
                     "c_acctbal    DOUBLE NOT NULL, "
                     "c_mktsegment CHAR(10)      NOT NULL, "
                     "c_comment    VARCHAR(117)  NOT NULL, "
                     "PRIMARY KEY (c_custkey) "
                     "); ");

    db->parse_schema("CREATE TABLE orders ( "
                     "o_orderkey      INTEGER       NOT NULL, "
                     "o_custkey       INTEGER       NOT NULL, "
                     "o_orderstatus   CHAR(1)       NOT NULL, "
                     // "o_totalprice    DECIMAL(15,2) NOT NULL, "
                     "o_totalprice    DOUBLE NOT NULL, "
                     "o_orderdate     DATE          NOT NULL, "
                     "o_orderpriority CHAR(15)      NOT NULL, "
                     "o_clerk         CHAR(15)      NOT NULL, "
                     "o_shippriority  INTEGER       NOT NULL, "
                     "o_comment       VARCHAR(79)   NOT NULL, "
                     "PRIMARY KEY (o_orderkey) "
                     "); ");

    db->parse_schema("CREATE TABLE lineitem ( "
                     "l_orderkey      INTEGER       NOT NULL, "
                     "l_partkey       INTEGER       NOT NULL, "
                     "l_suppkey       INTEGER       NOT NULL, "
                     "l_linenumber    INTEGER       NOT NULL, "
                     // "l_quantity      DECIMAL(15,2) NOT NULL, "
                     "l_quantity      DOUBLE        NOT NULL, "
                     // "l_extendedprice DECIMAL(15,2) NOT NULL, "
                     "l_extendedprice DOUBLE        NOT NULL, "
                     // "l_discount      DECIMAL(15,2) NOT NULL, "
                     "l_discount      DOUBLE        NOT NULL, "
                     // "l_tax           DECIMAL(15,2) NOT NULL, "
                     "l_tax           DOUBLE        NOT NULL, "
                     "l_returnflag    CHAR(1)       NOT NULL, "
                     "l_linestatus    CHAR(1)       NOT NULL, "
                     "l_shipdate      DATE          NOT NULL, "
                     "l_commitdate    DATE          NOT NULL, "
                     "l_receiptdate   DATE          NOT NULL, "
                     "l_shipinstruct  CHAR(25)      NOT NULL, "
                     "l_shipmode      CHAR(10)      NOT NULL, "
                     "l_comment       VARCHAR(44)   NOT NULL, "
                     "PRIMARY KEY (l_orderkey, l_linenumber) "
                     "); ");
}

TEST(sql, SelectFrom) {
    std::string path = PROJECT_SOURCE_DIR;
    path += "/tests/tpch/csv";
    std::cout << "Path: " << path << std::endl;

    auto catalogue = make_catalogue(path);
    auto db        = make_database(catalogue);
    set_tpch_schemas(db);

    auto table = db->query("SELECT * FROM customer;");

    EXPECT_TRUE(table);

    std::cout << table->to_string() << std::endl;
}

TEST(sql, HashJoin) {
    std::string path = PROJECT_SOURCE_DIR;
    path += "/tests/tpch/csv";
    std::cout << "Path: " << path << std::endl;

    auto catalogue = make_catalogue(path);
    auto db        = make_database(catalogue);
    set_tpch_schemas(db);

    auto table = db->query(
        "SELECT * FROM customer INNER JOIN orders ON customer.c_custkey = orders.o_custkey;");

    EXPECT_TRUE(table);

    if (table) std::cout << table->to_string() << std::endl;
}

TEST(sql, Multicolumn2HashJoin) {
    std::string path = PROJECT_SOURCE_DIR;
    path += "/tests/tpch/csv";
    std::cout << "Path: " << path << std::endl;

    auto catalogue = make_catalogue(path);
    auto db        = make_database(catalogue);
    set_tpch_schemas(db);

    auto table = db->query("SELECT * FROM customer INNER JOIN orders ON customer.c_custkey = "
                           "orders.o_custkey AND customer.c_name = orders.o_comment;");

    EXPECT_FALSE(table);

    // std::cout << table->to_string() << std::endl;
}

TEST(tpch, q1) {
    std::string path = PROJECT_SOURCE_DIR;
    path += "/tests/tpch/csv";
    std::cout << "Path: " << path << std::endl;
    auto catalogue = make_catalogue(path);
    auto db        = make_database(catalogue);
    set_tpch_schemas(db);

    auto table =
        db->query("SELECT "
                  "lineitem.l_returnflag, "
                  "lineitem.l_linestatus, "
                  "sum(lineitem.l_quantity) AS sum_qty, "
                  "sum(lineitem.l_extendedprice) AS sum_base_price, "
                  "sum(lineitem.l_extendedprice * (1 - lineitem.l_discount)) AS sum_disc_price, "
                  "sum(lineitem.l_extendedprice * (1 - lineitem.l_discount) * (1 + "
                  "lineitem.l_tax)) AS sum_charge, "
                  "avg(lineitem.l_quantity) AS avg_qty, "
                  "avg(lineitem.l_extendedprice) AS avg_price, "
                  "avg(lineitem.l_discount) AS avg_disc, "
                  "count(*) AS count_order "
                  "FROM "
                  "lineitem "
                  "WHERE lineitem.l_shipdate <= date '1998-09-02'"
                  "GROUP BY "
                  "lineitem.l_returnflag, "
                  "lineitem.l_linestatus "
                  "ORDER BY "
                  "lineitem.l_returnflag, "
                  "lineitem.l_linestatus ");
    EXPECT_TRUE(table);
    std::cout << table->to_string() << std::endl;
}

TEST(tpch, q2) {
    std::string path = PROJECT_SOURCE_DIR;
    path += "/tests/tpch/csv";
    std::cout << "Path: " << path << std::endl;
    auto catalogue = make_catalogue(path);
    auto db        = make_database(catalogue);
    set_tpch_schemas(db);

    auto table = db->query(
        "SELECT "
        "supplier.s_acctbal,"
        "supplier.s_name,"
        "nation.n_name,"
        "part.p_partkey,"
        "part.p_mfgr,"
        "supplier.s_address,"
        "supplier.s_phone,"
        "supplier.s_comment "
        "FROM "
        "part, supplier, partsupp, nation, region "
        "WHERE "
        "part.p_partkey = partsupp.ps_partkey "
        // uncomment the following 2 lines to have the original query. this is for having a result at output.
        // "AND part.p_size = 15 "
        // "AND part.p_type LIKE '%BRASS' "
        "AND region.r_name = 'EUROPE' "
        "AND supplier.s_suppkey = partsupp.ps_suppkey "
        "AND supplier.s_nationkey = nation.n_nationkey "
        "AND nation.n_regionkey = region.r_regionkey "
        "AND partsupp.ps_supplycost = 5.16 "
        "AND partsupp.ps_supplycost = ("
        "SELECT "
        "min(partsupp.ps_supplycost) "
        "FROM "
        "partsupp, supplier, nation, region "
        "WHERE "
        "part.p_partkey = partsupp.ps_partkey "
        "AND supplier.s_suppkey = partsupp.ps_suppkey "
        "AND supplier.s_nationkey = nation.n_nationkey "
        "AND nation.n_regionkey = region.r_regionkey "
        "AND region.r_name = 'EUROPE' "
        ")"
        "ORDER BY "
        "supplier.s_acctbal DESC,"
        "nation.n_name,"
        "supplier.s_name,"
        "part.p_partkey "
        "LIMIT 100;");
    EXPECT_TRUE(table);
    std::cout << table->to_string() << std::endl;
}

TEST(tpch, q3) {
    std::string path = PROJECT_SOURCE_DIR;
    path += "/tests/tpch/csv";
    std::cout << "Path: " << path << std::endl;
    auto catalogue = make_catalogue(path);
    auto db        = make_database(catalogue);
    set_tpch_schemas(db);

    auto table = db->query("SELECT "
                           "lineitem.l_orderkey, "
                           "sum(lineitem.l_extendedprice * (1 - lineitem.l_discount)) AS revenue, "
                           "orders.o_orderdate, "
                           "orders.o_shippriority "
                           "FROM "
                           "customer, orders, lineitem "
                           "WHERE "
                           "customer.c_mktsegment = 'BUILDING' "
                           "AND customer.c_custkey = orders.o_custkey "
                           "AND lineitem.l_orderkey = orders.o_orderkey "
                           "AND orders.o_orderdate < CAST('1995-03-15' AS date) "
                           "AND lineitem.l_shipdate > CAST('1995-03-15' AS date) "
                           "GROUP BY "
                           "lineitem.l_orderkey, "
                           "orders.o_orderdate, "
                           "orders.o_shippriority "
                           "ORDER BY "
                           "revenue DESC, "
                           "orders.o_orderdate "
                           "LIMIT 10; ");
    EXPECT_TRUE(table);
    std::cout << table->to_string() << std::endl;
}

TEST(tpch, q4) {
    std::string path = PROJECT_SOURCE_DIR;
    path += "/tests/tpch/csv";
    std::cout << "Path: " << path << std::endl;
    auto catalogue = make_catalogue(path);
    auto db        = make_database(catalogue);
    set_tpch_schemas(db);
    auto table = db->query("SELECT "
                           "orders.o_orderpriority, "
                           "count(*) AS order_count "
                           "FROM "
                           "orders "
                           "WHERE "
                           "orders.o_orderdate >= CAST('1993-07-01' AS date) "
                           "AND orders.o_orderdate < CAST('1993-10-01' AS date) "
                           "AND EXISTS ( "
                           "SELECT "
                           "* "
                           "FROM "
                           "lineitem "
                           "WHERE "
                           "lineitem.l_orderkey = orders.o_orderkey "
                           "AND lineitem.l_commitdate < lineitem.l_receiptdate "
                           ") "
                           "GROUP BY "
                           "orders.o_orderpriority "
                           "ORDER BY "
                           "orders.o_orderpriority ");
    EXPECT_TRUE(table);
    std::cout << table->to_string() << std::endl;
}

TEST(tpch, q5) {
    std::string path = PROJECT_SOURCE_DIR;
    path += "/tests/tpch/csv";
    std::cout << "Path: " << path << std::endl;
    auto catalogue = make_catalogue(path);
    auto db        = make_database(catalogue);
    set_tpch_schemas(db);

    auto table = db->query(
        "SELECT "
        "nation.n_name, "
        "sum(lineitem.l_extendedprice * (1 - lineitem.l_discount)) AS revenue "
        "FROM "
        "customer, orders, lineitem, supplier, nation, region "
        "WHERE "
        "customer.c_custkey = orders.o_custkey "
        "AND lineitem.l_orderkey = orders.o_orderkey "
        "AND lineitem.l_suppkey = supplier.s_suppkey "
        "AND customer.c_nationkey = supplier.s_nationkey "
        "AND supplier.s_nationkey = nation.n_nationkey "
        "AND nation.n_regionkey = region.r_regionkey "
        // uncomment the following line to have the original query. This is for having a result at output.
        // "AND region.r_name = 'ASIA' "
        "AND orders.o_orderdate >= CAST('1994-01-01' AS date) "
        "AND orders.o_orderdate < CAST('1995-01-01' AS date) "
        "GROUP BY "
        "nation.n_name "
        "ORDER BY "
        "revenue DESC ");
    EXPECT_TRUE(table);
    std::cout << table->to_string() << std::endl;
}

TEST(tpch, q6) {
    std::string path = PROJECT_SOURCE_DIR;
    path += "/tests/tpch/csv";
    std::cout << "Path: " << path << std::endl;
    auto catalogue = make_catalogue(path);
    auto db        = make_database(catalogue);
    set_tpch_schemas(db);

    auto table = db->query("SELECT "
                           "sum(lineitem.l_extendedprice * lineitem.l_discount) AS revenue "
                           "FROM "
                           "lineitem "
                           "WHERE "
                           "lineitem.l_shipdate >= CAST('1994-01-01' AS date) "
                           "AND lineitem.l_shipdate < CAST('1995-01-01' AS date) "
                           "AND lineitem.l_discount BETWEEN 0.05 AND 0.07 "
                           "AND lineitem.l_quantity < 24 ");
    EXPECT_TRUE(table);
    std::cout << table->to_string() << std::endl;
}

TEST(tpch, q7) {
    std::string path = PROJECT_SOURCE_DIR;
    path += "/tests/tpch/csv";
    std::cout << "Path: " << path << std::endl;
    auto catalogue = make_catalogue(path);
    auto db        = make_database(catalogue);
    set_tpch_schemas(db);

    auto table = db->query(
        "SELECT "
        "shipping.supp_nation, "
        "shipping.cust_nation, "
        "shipping.l_year, "
        "sum(shipping.volume) AS revenue "
        "FROM ( "
        "SELECT "
        "n1.n_name AS supp_nation, "
        "n2.n_name AS cust_nation, "
        "extract(year FROM lineitem.l_shipdate) AS l_year, "
        "lineitem.l_extendedprice, "
        "lineitem.l_extendedprice * (1 - lineitem.l_discount) AS volume "
        "FROM "
        "supplier, lineitem, orders, customer, nation n1, nation n2 "
        "WHERE "
        "supplier.s_suppkey = lineitem.l_suppkey "
        "AND orders.o_orderkey = lineitem.l_orderkey "
        "AND customer.c_custkey = orders.o_custkey "
        "AND supplier.s_nationkey = n1.n_nationkey "
        "AND customer.c_nationkey = n2.n_nationkey "
        // change GERAMANY and PERU to the desired value. They are selected so that output be non empty.
        "AND ((n2.n_name = 'GERMANY' AND n1.n_name = 'PERU') OR (n2.n_name = 'PERU' AND n1.n_name "
        "= 'GERMANY')) "
        "AND lineitem.l_shipdate BETWEEN CAST('1995-01-01' AS date) AND CAST('1996-12-31' AS date) "
        ") AS shipping "
        "GROUP BY "
        "shipping.supp_nation, "
        "shipping.cust_nation, "
        "shipping.l_year "
        "ORDER BY "
        "shipping.supp_nation, "
        "shipping.cust_nation, "
        "shipping.l_year ");
    EXPECT_TRUE(table);
    std::cout << table->to_string() << std::endl;
}

TEST(tpch, q8) {
    std::string path = PROJECT_SOURCE_DIR;
    path += "/tests/tpch/csv";
    std::cout << "Path: " << path << std::endl;
    auto catalogue = make_catalogue(path);
    auto db        = make_database(catalogue);
    set_tpch_schemas(db);

    auto table = db->query(
        "SELECT "
        "all_nations.o_year, "
        "sum( "
        "CASE WHEN all_nations.nation = 'BRAZIL' THEN "
        "all_nations.volume "
        "ELSE "
        "0 "
        "END"
        ") / sum(all_nations.volume) AS mkt_share, "
        "sum(all_nations.volume) AS sum_volume "
        "FROM ( "
        "SELECT "
        "extract(year FROM orders.o_orderdate) AS o_year, "
        "lineitem.l_extendedprice * (1 - lineitem.l_discount) AS volume, "
        "n2.n_name AS nation "
        "FROM "
        "part, supplier, lineitem, orders, customer, nation n1, nation n2, region "
        "WHERE "
        "part.p_partkey = lineitem.l_partkey "
        "AND supplier.s_suppkey = lineitem.l_suppkey "
        "AND lineitem.l_orderkey = orders.o_orderkey "
        "AND orders.o_custkey = customer.c_custkey "
        "AND customer.c_nationkey = n1.n_nationkey "
        "AND n1.n_regionkey = region.r_regionkey "
        "AND region.r_name = 'AMERICA' "
        "AND supplier.s_nationkey = n2.n_nationkey "
        "AND orders.o_orderdate BETWEEN CAST('1995-01-01' AS date) AND CAST('1996-12-31' AS date) "
        "AND part.p_type = 'ECONOMY ANODIZED STEEL' "
        ") AS all_nations "
        "GROUP BY "
        "all_nations.o_year "
        "ORDER BY "
        "all_nations.o_year ");
    EXPECT_TRUE(table);
    std::cout << table->to_string() << std::endl;
}

TEST(tpch, q9) {
    std::string path = PROJECT_SOURCE_DIR;
    path += "/tests/tpch/csv";
    std::cout << "Path: " << path << std::endl;
    auto catalogue = make_catalogue(path);
    auto db        = make_database(catalogue);
    set_tpch_schemas(db);

    auto table = db->query("SELECT "
                           "profit.nation, "
                           "profit.o_year, "
                           "sum(profit.amount) AS sum_profit "
                           "FROM ( "
                           "SELECT "
                           "nation.n_name AS nation, "
                           "extract(year FROM orders.o_orderdate) AS o_year, "
                           "lineitem.l_extendedprice * (1 - lineitem.l_discount) - "
                           "partsupp.ps_supplycost * lineitem.l_quantity AS amount "
                           "FROM "
                           "part, supplier, lineitem, partsupp, orders, nation "
                           "WHERE "
                           "supplier.s_suppkey = lineitem.l_suppkey "
                           "AND partsupp.ps_suppkey = lineitem.l_suppkey "
                           "AND partsupp.ps_partkey = lineitem.l_partkey "
                           "AND part.p_partkey = lineitem.l_partkey "
                           "AND orders.o_orderkey = lineitem.l_orderkey "
                           "AND supplier.s_nationkey = nation.n_nationkey "
                           "AND part.p_name LIKE '%green%' "
                           ") AS profit "
                           "GROUP BY "
                           "profit.nation, "
                           "profit.o_year "
                           "ORDER BY "
                           "profit.nation, "
                           "profit.o_year DESC; ");
    EXPECT_TRUE(table);
    std::cout << table->to_string() << std::endl;
}

TEST(tpch, q10) {
    std::string path = PROJECT_SOURCE_DIR;
    path += "/tests/tpch/csv";
    std::cout << "Path: " << path << std::endl;
    auto catalogue = make_catalogue(path);
    auto db        = make_database(catalogue);
    set_tpch_schemas(db);

    auto table = db->query("SELECT "
                           "customer.c_custkey, "
                           "customer.c_name, "
                           "sum(lineitem.l_extendedprice * (1 - lineitem.l_discount)) AS revenue, "
                           "customer.c_acctbal, "
                           "nation.n_name, "
                           "customer.c_address, "
                           "customer.c_phone, "
                           "customer.c_comment "
                           "FROM "
                           "customer, orders, lineitem, nation "
                           "WHERE "
                           "customer.c_custkey = orders.o_custkey "
                           "AND lineitem.l_orderkey = orders.o_orderkey "
                           "AND orders.o_orderdate >= CAST('1993-10-01' AS date) "
                           "AND orders.o_orderdate < CAST('1994-01-01' AS date) "
                           "AND lineitem.l_returnflag = 'R' "
                           "AND customer.c_nationkey = nation.n_nationkey "
                           "GROUP BY "
                           "customer.c_custkey, "
                           "customer.c_name, "
                           "customer.c_acctbal, "
                           "customer.c_phone, "
                           "nation.n_name, "
                           "customer.c_address, "
                           "customer.c_comment "
                           "ORDER BY "
                           "revenue DESC "
                           "LIMIT 20; ");
    EXPECT_TRUE(table);
    std::cout << table->to_string() << std::endl;
}

TEST(tpch, q11) {
    std::string path = PROJECT_SOURCE_DIR;
    path += "/tests/tpch/csv";
    std::cout << "Path: " << path << std::endl;
    auto catalogue = make_catalogue(path);
    auto db        = make_database(catalogue);
    set_tpch_schemas(db);

    auto table = db->query("SELECT "
                           "ps.ps_partkey, "
                           "sum(ps.ps_supplycost * ps.ps_availqty) AS value "
                           "FROM "
                           "partsupp ps, "
                           "supplier s, "
                           "nation n "
                           "WHERE "
                           "ps.ps_suppkey = s.s_suppkey "
                           "AND s.s_nationkey = n.n_nationkey "
                           "AND n.n_name = 'IRAN' "
                           "GROUP BY "
                           "ps.ps_partkey "
                           "HAVING "
                           "sum(ps.ps_supplycost * ps.ps_availqty) > ( "
                           "SELECT "
                           "sum(ps.ps_supplycost * ps.ps_availqty) * 0.0001000000 "
                           "FROM "
                           "partsupp ps, "
                           "supplier s, "
                           "nation n "
                           "WHERE "
                           "ps.ps_suppkey = s.s_suppkey "
                           "AND s.s_nationkey = n.n_nationkey "
                           "AND n.n_name = 'IRAN' "
                           ") "
                           "ORDER BY "
                           "value DESC; ");
    EXPECT_TRUE(table);
    std::cout << table->to_string() << std::endl;
}


TEST(tpch, q12) {
    std::string path = PROJECT_SOURCE_DIR;
    path += "/tests/tpch/csv";
    std::cout << "Path: " << path << std::endl;
    auto catalogue = make_catalogue(path);
    auto db        = make_database(catalogue);
    set_tpch_schemas(db);

    auto table = db->query("SELECT "
                           "lineitem.l_shipmode, "
                           "sum( "
                           "CASE WHEN orders.o_orderpriority = '1-URGENT' "
                           "OR orders.o_orderpriority = '2-HIGH' THEN 1 "
                           "ELSE "
                           "0 "
                           "END) AS high_line_count, "
                           "sum( "
                           "CASE WHEN orders.o_orderpriority <> '1-URGENT' "
                           "AND orders.o_orderpriority <> '2-HIGH' THEN 1 "
                           "ELSE "
                           "0 "
                           "END) AS low_line_count "
                           "FROM "
                           "orders, lineitem "
                           "WHERE "
                           "orders.o_orderkey = lineitem.l_orderkey "
                           "AND lineitem.l_shipmode IN ('MAIL', 'SHIP') "
                           "AND lineitem.l_commitdate < lineitem.l_receiptdate "
                           "AND lineitem.l_shipdate < lineitem.l_commitdate "
                           "AND lineitem.l_receiptdate >= CAST('1994-01-01' AS date) "
                           "AND lineitem.l_receiptdate < CAST('1995-01-01' AS date) "
                           "GROUP BY "
                           "lineitem.l_shipmode "
                           "ORDER BY "
                           "lineitem.l_shipmode ");
    EXPECT_TRUE(table);
    std::cout << table->to_string() << std::endl;
}

TEST(tpch, q13) {
    std::string path = PROJECT_SOURCE_DIR;
    path += "/tests/tpch/csv";
    std::cout << "Path: " << path << std::endl;
    auto catalogue = make_catalogue(path);
    auto db        = make_database(catalogue);
    set_tpch_schemas(db);

    auto table = db->query("SELECT "
                           "c_orders.c_count, "
                           "count(*) AS custdist "
                           "FROM ( "
                           "SELECT "
                           "customer.c_custkey, "
                           "count(orders.o_orderkey) "
                           "FROM "
                           "customer LEFT OUTER JOIN orders ON "
                           "customer.c_custkey = orders.o_custkey "
                           "AND orders.o_comment NOT LIKE '%special%requests%' "
                           "GROUP BY "
                           "customer.c_custkey "
                           ") AS c_orders (c_custkey, c_count) "
                           "GROUP BY "
                           "c_orders.c_count "
                           "ORDER BY "
                           "c_orders.custdist DESC, "
                           "c_orders.c_count DESC ");
    EXPECT_TRUE(table);
    std::cout << table->to_string() << std::endl;
}

TEST(tpch, q14) {
    std::string path = PROJECT_SOURCE_DIR;
    path += "/tests/tpch/csv";
    std::cout << "Path: " << path << std::endl;
    auto catalogue = make_catalogue(path);
    auto db        = make_database(catalogue);
    set_tpch_schemas(db);

    auto table = db->query(
        "SELECT "
        "100.00 * sum( "
        "CASE WHEN part.p_type LIKE 'PROMO%' THEN "
        "lineitem.l_extendedprice * (1 - lineitem.l_discount) "
        "ELSE "
        "0 "
        "END) / sum(lineitem.l_extendedprice * (1 - lineitem.l_discount)) AS promo_revenue "
        "FROM "
        "lineitem, part "
        "WHERE "
        "lineitem.l_partkey = part.p_partkey "
        "AND lineitem.l_shipdate >= CAST('1995-09-01' AS date) "
        "AND lineitem.l_shipdate < CAST('1995-10-01' AS date); ");
    EXPECT_TRUE(table);
    std::cout << table->to_string() << std::endl;
}

TEST(tpch, q15) {
    std::string path = PROJECT_SOURCE_DIR;
    path += "/tests/tpch/csv";
    std::cout << "Path: " << path << std::endl;
    auto catalogue = make_catalogue(path);
    auto db        = make_database(catalogue);
    set_tpch_schemas(db);

    auto table =
        db->query("SELECT "
                  "supplier.s_suppkey, "
                  "supplier.s_name, "
                  "supplier.s_address, "
                  "supplier.s_phone, "
                  "revenue0.total_revenue "
                  "FROM "
                  "supplier, "
                  "( "
                  "SELECT "
                  "lineitem.l_suppkey AS supplier_no, "
                  "sum(lineitem.l_extendedprice * (1 - lineitem.l_discount)) AS total_revenue "
                  "FROM "
                  "lineitem "
                  "WHERE "
                  "lineitem.l_shipdate >= CAST('1996-01-01' AS date) "
                  "AND lineitem.l_shipdate < CAST('1996-04-01' AS date) "
                  "GROUP BY "
                  "lineitem.l_suppkey) revenue0 "
                  "WHERE "
                  "supplier.s_suppkey = revenue0.supplier_no "
                  "AND revenue0.total_revenue = ( "
                  "SELECT "
                  "max(revenue1.total_revenue) "
                  "FROM ( "
                  "SELECT "
                  "lineitem.l_suppkey AS supplier_no, "
                  "sum(lineitem.l_extendedprice * (1 - lineitem.l_discount)) AS total_revenue "
                  "FROM "
                  "lineitem.lineitem "
                  "WHERE "
                  "lineitem.l_shipdate >= CAST('1996-01-01' AS date) "
                  "AND lineitem.l_shipdate < CAST('1996-04-01' AS date) "
                  "GROUP BY "
                  "lineitem.l_suppkey) revenue1) "
                  "ORDER BY "
                  "supplier.s_suppkey "
                  "LIMIT 10 ");
    EXPECT_TRUE(table);
    std::cout << table->to_string() << std::endl;
}

TEST(tpch, q16) {
    std::string path = PROJECT_SOURCE_DIR;
    path += "/tests/tpch/csv";
    std::cout << "Path: " << path << std::endl;
    auto catalogue = make_catalogue(path);
    auto db        = make_database(catalogue);
    set_tpch_schemas(db);

    auto table = db->query("SELECT "
                           "part.p_brand, "
                           "part.p_type, "
                           "part.p_size, "
                           "count(DISTINCT partsupp.ps_suppkey) AS supplier_cnt, "
                           "count(partsupp.ps_suppkey) AS supplier_cnt2 "
                           "FROM "
                           "partsupp, part "
                           "WHERE "
                           "part.p_partkey = partsupp.ps_partkey "
                           "AND part.p_brand <> 'Brand#45' "
                           "AND part.p_type NOT LIKE 'MEDIUM POLISHED%' "
                           "AND part.p_size IN (49, 14, 23, 45, 19, 3, 36, 9) "
                           "AND partsupp.ps_suppkey NOT IN ( "
                           "SELECT "
                           "supplier.s_suppkey "
                           "FROM "
                           "supplier "
                           "WHERE "
                           "supplier.s_comment LIKE '%Customer%Complaints%') "
                           "GROUP BY "
                           "part.p_brand, "
                           "part.p_type, "
                           "part.p_size "
                           "ORDER BY "
                           "supplier_cnt DESC, "
                           "part.p_brand, "
                           "part.p_type, "
                           "part.p_size ");
    EXPECT_TRUE(table);
    std::cout << table->to_string() << std::endl;
}

TEST(tpch, q17) {
    std::string path = PROJECT_SOURCE_DIR;
    path += "/tests/tpch/csv";
    std::cout << "Path: " << path << std::endl;
    auto catalogue = make_catalogue(path);
    auto db        = make_database(catalogue);
    set_tpch_schemas(db);

    auto table = db->query(
        "SELECT "
        "sum(lineitem.l_extendedprice) / 7.0 AS avg_yearly "
        "FROM "
        "lineitem, part "
        "WHERE "
        "part.p_partkey = lineitem.l_partkey "
        "AND part.p_brand = 'Brand#23' "
        // uncomment the following line to have the original query. This is for having a result at output.
        // "AND part.p_container = 'MED BOX' "
        "AND lineitem.l_quantity < ( "
        "SELECT "
        "0.2 * avg(lineitem.l_quantity) "
        "FROM "
        "lineitem "
        "WHERE "
        "lineitem.l_partkey = part.p_partkey) ");
    EXPECT_TRUE(table);
    std::cout << table->to_string() << std::endl;
}


TEST(tpch, q18) {
    std::string path = PROJECT_SOURCE_DIR;
    path += "/tests/tpch/csv";
    std::cout << "Path: " << path << std::endl;
    auto catalogue = make_catalogue(path);
    auto db        = make_database(catalogue);
    set_tpch_schemas(db);

    auto table = db->query("SELECT "
                           "customer.c_name, "
                           "customer.c_custkey, "
                           "orders.o_orderkey, "
                           "orders.o_orderdate, "
                           "orders.o_totalprice, "
                           "sum(lineitem.l_quantity) "
                           "FROM "
                           "customer, orders, lineitem "
                           "WHERE "
                           "orders.o_orderkey IN ( "
                           "SELECT "
                           "lineitem.l_orderkey "
                           "FROM "
                           "lineitem "
                           "GROUP BY "
                           "lineitem.l_orderkey "
                           "HAVING "
                           // change 30 to 300 to have the original query.
                           "sum(lineitem.l_quantity) > 30) "
                           "AND customer.c_custkey = orders.o_custkey "
                           "AND orders.o_orderkey = lineitem.l_orderkey "
                           "GROUP BY "
                           "customer.c_name, "
                           "customer.c_custkey, "
                           "orders.o_orderkey, "
                           "orders.o_orderdate, "
                           "orders.o_totalprice "
                           "ORDER BY "
                           "orders.o_totalprice DESC, "
                           "orders.o_orderdate "
                           "LIMIT 100 ");
    EXPECT_TRUE(table);
    std::cout << table->to_string() << std::endl;
}

TEST(tpch, q19) {
    std::string path = PROJECT_SOURCE_DIR;
    path += "/tests/tpch/csv";
    std::cout << "Path: " << path << std::endl;
    auto catalogue = make_catalogue(path);
    auto db        = make_database(catalogue);
    set_tpch_schemas(db);

    auto table = db->query(
        // uncommnet the commentedt lines to get the original query. This if for having a result at output.
        "SELECT "
        "sum(lineitem.l_extendedprice * (1 - lineitem.l_discount) ) as revenue "
        "FROM "
        "lineitem, "
        "part "
        "WHERE "
        "( "
        "part.p_partkey = lineitem.l_partkey "
        // "AND part.p_brand = 'Brand#12' "
        "AND part.p_container in ( 'SM CASE', 'SM BOX', 'SM PACK', 'SM PKG') "
        "AND lineitem.l_quantity >= 1 AND lineitem.l_quantity <= 1 + 10 "
        "AND part.p_size between 1 AND 5 "
        "AND lineitem.l_shipmode in ('AIR', 'AIR REG') "
        "AND lineitem.l_shipinstruct = 'DELIVER IN PERSON' "
        ") OR "
        "( "
        "part.p_partkey = lineitem.l_partkey "
        // "AND part.p_brand = 'Brand#23' "
        "AND part.p_container in ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK') "
        "AND lineitem.l_quantity >= 10 AND lineitem.l_quantity <= 10 + 10 "
        "AND part.p_size between 1 AND 10 "
        "AND lineitem.l_shipmode in ('AIR', 'AIR REG') "
        "AND lineitem.l_shipinstruct = 'DELIVER IN PERSON' "
        ") OR "
        "( "
        "part.p_partkey = lineitem.l_partkey "
        // "AND part.p_brand = 'Brand#34' "
        "AND part.p_container in ( 'LG CASE', 'LG BOX', 'LG PACK', 'LG PKG') "
        "AND lineitem.l_quantity >= 20 AND lineitem.l_quantity <= 20 + 10 "
        "AND part.p_size between 1 AND 15 "
        "AND lineitem.l_shipmode in ('AIR', 'AIR REG') "
        "AND lineitem.l_shipinstruct = 'DELIVER IN PERSON' "
        ") ");
    EXPECT_TRUE(table);
    std::cout << table->to_string() << std::endl;
}

TEST(tpch, q20) {
    std::string path = PROJECT_SOURCE_DIR;
    path += "/tests/tpch/csv";
    std::cout << "Path: " << path << std::endl;
    auto catalogue = make_catalogue(path);
    auto db        = make_database(catalogue);
    set_tpch_schemas(db);

    auto table = db->query("SELECT "
                           "supplier.s_name, "
                           "supplier.s_address, "
                           "nation.n_name "
                           "FROM "
                           "supplier, nation "
                           "WHERE "
                           "supplier.s_suppkey IN ( "
                           "SELECT "
                           "partsupp.ps_suppkey "
                           "FROM "
                           "partsupp "
                           "WHERE "
                           "partsupp.ps_partkey IN ( "
                           "SELECT "
                           "part.p_partkey "
                           "FROM "
                           "part "
                           "WHERE "
                           "part.p_name like 'forest%' "
                           ") "
                           "AND partsupp.ps_availqty > ( "
                           "SELECT "
                           // change 0.2 to 0.5, this is just for having output
                           "0.2 * sum(lineitem.l_quantity) "
                           "FROM "
                           "lineitem "
                           "WHERE "
                           "lineitem.l_partkey = partsupp.ps_partkey "
                           "AND lineitem.l_suppkey = partsupp.ps_suppkey "
                           "AND lineitem.l_shipdate >= CAST('1994-01-01' AS date) "
                           "AND lineitem.l_shipdate < CAST('1995-01-01' AS date) "
                           ") "
                           ") "
                           "AND supplier.s_nationkey = nation.n_nationkey "
                           // change peru to canada, this is just for having output
                           "AND nation.n_name = 'PERU' "
                           "ORDER BY "
                           "supplier.s_name ");
    EXPECT_TRUE(table);
    std::cout << table->to_string() << std::endl;
}

TEST(tpch, q21) {
    std::string path = PROJECT_SOURCE_DIR;
    path += "/tests/tpch/csv";
    std::cout << "Path: " << path << std::endl;
    auto catalogue = make_catalogue(path);
    auto db        = make_database(catalogue);
    set_tpch_schemas(db);

    auto table = db->query("SELECT "
                           "supplier.s_name, "
                           "count(*) AS numwait "
                           "FROM "
                           "supplier, lineitem l1, orders, nation "
                           "WHERE "
                           "supplier.s_suppkey = l1.l_suppkey "
                           "AND orders.o_orderkey = l1.l_orderkey "
                           "AND orders.o_orderstatus = 'F' "
                           "AND l1.l_receiptdate > l1.l_commitdate "
                           "AND EXISTS ( "
                           "SELECT "
                           "* "
                           "FROM "
                           "lineitem l2 "
                           "WHERE "
                           "l2.l_orderkey = l1.l_orderkey "
                           "AND l2.l_suppkey <> l1.l_suppkey) "
                           "AND NOT EXISTS ( "
                           "SELECT "
                           "* "
                           "FROM "
                           "lineitem l3 "
                           "WHERE "
                           "l3.l_orderkey = l1.l_orderkey "
                           "AND l3.l_suppkey <> l1.l_suppkey "
                           "AND l3.l_receiptdate > l3.l_commitdate) "
                           "AND supplier.s_nationkey = nation.n_nationkey "
                           "AND nation.n_name = 'SAUDI ARABIA' "
                           "GROUP BY "
                           "supplier.s_name "
                           "ORDER BY "
                           "numwait DESC, "
                           "supplier.s_name "
                           "LIMIT 100; ");
    EXPECT_FALSE(table);
    // std::cout << table->to_string() << std::endl;
}

TEST(tpch, q22) {
    std::string path = PROJECT_SOURCE_DIR;
    path += "/tests/tpch/csv";
    std::cout << "Path: " << path << std::endl;
    auto catalogue = make_catalogue(path);
    auto db        = make_database(catalogue);
    set_tpch_schemas(db);

    auto table = db->query(
        "SELECT "
        "custsale.cntrycode, "
        "count(*) AS numcust, "
        "sum(custsale.c_acctbal) AS totacctbal "
        "FROM ( "
        "SELECT "
        "substring(customer.c_phone, 1, 2) AS cntrycode, "
        "customer.c_acctbal "
        "FROM "
        "customer "
        "WHERE "
        "substring(customer.c_phone, 1, 2) IN ('13', '31', '23', '29', '30', '18', '17') "
        "AND customer.c_acctbal > ( "
        "SELECT "
        "avg(customer.c_acctbal) "
        "FROM "
        "customer "
        "WHERE "
        "customer.c_acctbal > 0.00 "
        "AND substring(customer.c_phone, 1, 2) IN ('13', '31', '23', '29', '30', '18', '17') "
        ") "
        "AND NOT EXISTS ( "
        "SELECT "
        "* "
        "FROM "
        "orders "
        "WHERE "
        "orders.o_custkey = customer.c_custkey) "
        ") AS custsale "
        "GROUP BY "
        "custsale.cntrycode "
        "ORDER BY "
        "custsale.cntrycode ");
    EXPECT_TRUE(table);
    std::cout << table->to_string() << std::endl;
}


TEST(sql, create) {
    std::string path = PROJECT_SOURCE_DIR;
    path += "/tests/tpch/csv";
    std::cout << "Path: " << path << std::endl;
    auto catalogue = make_catalogue(path);
    auto db        = make_database(catalogue);
    set_tpch_schemas(db);
}

}  // namespace test
