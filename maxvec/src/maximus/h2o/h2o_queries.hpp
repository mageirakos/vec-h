#pragma once

#include <maximus/dag/query_node.hpp>
#include <maximus/dag/query_plan.hpp>
#include <maximus/database.hpp>
#include <maximus/types/device_table_ptr.hpp>

namespace maximus {

namespace h2o {
std::shared_ptr<QueryPlan> q1(std::shared_ptr<Database>& db, DeviceType device = DeviceType::CPU);

std::shared_ptr<QueryPlan> q2(std::shared_ptr<Database>& db, DeviceType device = DeviceType::CPU);

std::shared_ptr<QueryPlan> q3(std::shared_ptr<Database>& db, DeviceType device = DeviceType::CPU);

std::shared_ptr<QueryPlan> q4(std::shared_ptr<Database>& db, DeviceType device = DeviceType::CPU);

std::shared_ptr<QueryPlan> q5(std::shared_ptr<Database>& db, DeviceType device = DeviceType::CPU);

std::shared_ptr<QueryPlan> q6(std::shared_ptr<Database>& db, DeviceType device = DeviceType::CPU);

std::shared_ptr<QueryPlan> q7(std::shared_ptr<Database>& db, DeviceType device = DeviceType::CPU);

std::shared_ptr<QueryPlan> q8(std::shared_ptr<Database>& db, DeviceType device = DeviceType::CPU);

std::shared_ptr<QueryPlan> q9(std::shared_ptr<Database>& db, DeviceType device = DeviceType::CPU);

std::shared_ptr<QueryPlan> q10(std::shared_ptr<Database>& db, DeviceType device = DeviceType::CPU);

std::shared_ptr<QueryPlan> query_plan(const std::string& query,
                                      std::shared_ptr<Database>& db,
                                      DeviceType device = DeviceType::CPU);

std::shared_ptr<Schema> schema(const std::string& table_name);

std::vector<std::string> table_names();

std::vector<std::shared_ptr<Schema>> schemas();

}  // namespace h2o
// namespace h2o
}  // namespace maximus
