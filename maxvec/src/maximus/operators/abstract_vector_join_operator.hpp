#pragma once

#include <maximus/operators/abstract_operator.hpp>
#include <maximus/operators/properties.hpp>

namespace maximus {

class AbstractVectorJoinOperator : public AbstractOperator {
public:
    AbstractVectorJoinOperator(std::shared_ptr<MaximusContext>& ctx,
                               std::vector<std::shared_ptr<maximus::Schema>> input_schemas,
                               std::shared_ptr<VectorJoinProperties> properties);

    virtual int get_data_port() {
        return 0;
    }

    virtual int get_query_port() {
        return 1;
    }

protected:
    std::shared_ptr<VectorJoinProperties> abstract_properties;
};

}  // namespace maximus
