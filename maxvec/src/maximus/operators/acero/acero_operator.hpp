#pragma once

#include <maximus/operators/acero/proxy_operator.hpp>

namespace maximus::acero {
class AceroOperator {
public:
    AceroOperator() = default;

    std::unique_ptr<ProxyOperator> proxy_operator;
};
}  // namespace maximus::acero