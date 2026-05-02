#pragma once
#include <iostream>
#include <maximus/context.hpp>
#include <maximus/database_catalogue.hpp>
#include <maximus/exec/executor.hpp>
#include <maximus/sql/parser.hpp>
#include <maximus/types/table.hpp>
#include <string>

namespace maximus {

class Database {
public:
    Database();

    explicit Database(std::shared_ptr<DatabaseCatalogue> db_catalogue,
                      std::shared_ptr<MaximusContext> ctx = make_context());

    // immediate execution of the query plan
    // equivalent to scheduling and then executing the query plan
    TablePtr query(std::string sql_query);
    TablePtr query(std::shared_ptr<QueryPlan>& query_plan);

    void parse_schema(std::string sql_query);

    void schedule(std::shared_ptr<QueryPlan>& query_plan);

    std::vector<TablePtr> execute();

    std::shared_ptr<MaximusContext>& get_context();
    [[nodiscard]] const std::shared_ptr<MaximusContext>& get_context() const;

    [[nodiscard]] const std::shared_ptr<DatabaseCatalogue>& get_catalogue() const;

    std::shared_ptr<DatabaseCatalogue>& get_catalogue();

    void load_table(std::string table_name,
                    const std::shared_ptr<Schema>& schema       = nullptr,
                    const std::vector<std::string>& field_names = {},
                    const DeviceType& storage_device            = DeviceType::CPU);

    [[nodiscard]] DeviceTablePtr get_table(std::string table_name);
    
    // Manually set/update a table in the database (e.g. after modification/move)
    void set_table(std::string table_name, DeviceTablePtr table, std::shared_ptr<Schema> schema);

    // Zero-copy / pass-by-reference variants
    void load_table_nocopy(std::string table_name,
                           const std::shared_ptr<Schema>& schema       = nullptr,
                           const std::vector<std::string>& field_names = {},
                           const DeviceType& storage_device            = DeviceType::CPU);

    // Zero-copy getter: returns stored DeviceTablePtr directly without cloning/copying
    [[nodiscard]] DeviceTablePtr get_table_nocopy(std::string table_name);

    std::vector<std::string> get_table_names();

private:
    std::string base_path_ = "./";
    std::shared_ptr<MaximusContext> ctx_;
    std::shared_ptr<DatabaseCatalogue> db_catalogue_;

    std::unordered_map<std::string, DeviceTablePtr> tables_;
    std::unordered_map<std::string, std::shared_ptr<Schema>> schemas_;

    std::shared_ptr<Executor> executor_;

    std::vector<std::shared_ptr<QueryPlan>> pending_qp_;
};

std::shared_ptr<Database> make_database(std::shared_ptr<DatabaseCatalogue> db_catalogue);

std::shared_ptr<Database> make_database(std::shared_ptr<DatabaseCatalogue> db_catalogue,
                                        std::shared_ptr<MaximusContext> ctx);

}  // namespace maximus
