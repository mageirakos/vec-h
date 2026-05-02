#pragma once

#include <maximus/dag/query_node.hpp>
#include <maximus/dag/query_plan.hpp>
#include <maximus/database.hpp>
#include <maximus/types/device_table_ptr.hpp>

namespace maximus {

namespace clickbench {
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

std::shared_ptr<QueryPlan> q11(std::shared_ptr<Database>& db, DeviceType device = DeviceType::CPU);

std::shared_ptr<QueryPlan> q12(std::shared_ptr<Database>& db, DeviceType device = DeviceType::CPU);

std::shared_ptr<QueryPlan> q13(std::shared_ptr<Database>& db, DeviceType device = DeviceType::CPU);

std::shared_ptr<QueryPlan> q14(std::shared_ptr<Database>& db, DeviceType device = DeviceType::CPU);

std::shared_ptr<QueryPlan> q15(std::shared_ptr<Database>& db, DeviceType device = DeviceType::CPU);

std::shared_ptr<QueryPlan> q16(std::shared_ptr<Database>& db, DeviceType device = DeviceType::CPU);

std::shared_ptr<QueryPlan> q17(std::shared_ptr<Database>& db, DeviceType device = DeviceType::CPU);

std::shared_ptr<QueryPlan> q18(std::shared_ptr<Database>& db, DeviceType device = DeviceType::CPU);

std::shared_ptr<QueryPlan> q19(std::shared_ptr<Database>& db, DeviceType device = DeviceType::CPU);

std::shared_ptr<QueryPlan> q20(std::shared_ptr<Database>& db, DeviceType device = DeviceType::CPU);

std::shared_ptr<QueryPlan> q21(std::shared_ptr<Database>& db, DeviceType device = DeviceType::CPU);

std::shared_ptr<QueryPlan> q22(std::shared_ptr<Database>& db, DeviceType device = DeviceType::CPU);

std::shared_ptr<QueryPlan> q23(std::shared_ptr<Database>& db, DeviceType device = DeviceType::CPU);

std::shared_ptr<QueryPlan> q24(std::shared_ptr<Database>& db, DeviceType device = DeviceType::CPU);

std::shared_ptr<QueryPlan> q25(std::shared_ptr<Database>& db, DeviceType device = DeviceType::CPU);

std::shared_ptr<QueryPlan> q26(std::shared_ptr<Database>& db, DeviceType device = DeviceType::CPU);

std::shared_ptr<QueryPlan> q27(std::shared_ptr<Database>& db, DeviceType device = DeviceType::CPU);

std::shared_ptr<QueryPlan> q28(std::shared_ptr<Database>& db, DeviceType device = DeviceType::CPU);

std::shared_ptr<QueryPlan> q29(std::shared_ptr<Database>& db, DeviceType device = DeviceType::CPU);

std::shared_ptr<QueryPlan> q30(std::shared_ptr<Database>& db, DeviceType device = DeviceType::CPU);

std::shared_ptr<QueryPlan> q31(std::shared_ptr<Database>& db, DeviceType device = DeviceType::CPU);

std::shared_ptr<QueryPlan> q32(std::shared_ptr<Database>& db, DeviceType device = DeviceType::CPU);

std::shared_ptr<QueryPlan> q33(std::shared_ptr<Database>& db, DeviceType device = DeviceType::CPU);

std::shared_ptr<QueryPlan> q34(std::shared_ptr<Database>& db, DeviceType device = DeviceType::CPU);

std::shared_ptr<QueryPlan> q35(std::shared_ptr<Database>& db, DeviceType device = DeviceType::CPU);

std::shared_ptr<QueryPlan> q36(std::shared_ptr<Database>& db, DeviceType device = DeviceType::CPU);

std::shared_ptr<QueryPlan> q37(std::shared_ptr<Database>& db, DeviceType device = DeviceType::CPU);

std::shared_ptr<QueryPlan> q38(std::shared_ptr<Database>& db, DeviceType device = DeviceType::CPU);

std::shared_ptr<QueryPlan> q40(std::shared_ptr<Database>& db, DeviceType device = DeviceType::CPU);

std::shared_ptr<QueryPlan> q41(std::shared_ptr<Database>& db, DeviceType device = DeviceType::CPU);

std::shared_ptr<QueryPlan> q42(std::shared_ptr<Database>& db, DeviceType device = DeviceType::CPU);

std::shared_ptr<QueryPlan> query_plan(const std::string& query,
                                      std::shared_ptr<Database>& db,
                                      DeviceType device = DeviceType::CPU);

std::shared_ptr<Schema> schema(const std::string& table_name);

std::vector<std::string> table_names();

std::vector<std::shared_ptr<Schema>> schemas();

}  // namespace clickbench
}  // namespace maximus
