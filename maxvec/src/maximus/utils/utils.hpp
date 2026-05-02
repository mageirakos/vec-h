#pragma once

#include <maximus/types/device_table_ptr.hpp>
#include <memory>

namespace maximus {

bool ends_with(const std::string& full_string, const std::string& ending);

bool starts_with(const std::string& full_string, const std::string& start);

bool contains(const std::string& full_string, const std::string& substring);

DeviceTablePtr read_table(std::shared_ptr<MaximusContext>& ctx,
                          std::string path,
                          const std::shared_ptr<Schema>& schema           = nullptr,
                          const std::vector<std::string>& include_columns = {},
                          const DeviceType& storage_device                = DeviceType::CPU);

DeviceTablePtr read_table_partitioned(std::shared_ptr<MaximusContext>& ctx,
                                      const std::vector<std::string>& paths,
                                      const std::shared_ptr<Schema>& schema           = nullptr,
                                      const std::vector<std::string>& include_columns = {},
                                      const DeviceType& storage_device                = DeviceType::CPU);

}  // namespace maximus
