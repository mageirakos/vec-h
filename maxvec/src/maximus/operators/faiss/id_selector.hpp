#pragma once
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/IDSelector.h>

#include <maximus/operators/abstract_operator.hpp>
#include <maximus/types/expression.hpp>

namespace maximus::faiss {

/************************************************************************************/
// IDSelector Helper Classes                                                         /
/************************************************************************************/

struct IDSelectorBitmap : ::faiss::IDSelectorBitmap {
    std::shared_ptr<arrow::BooleanArray> array;
    IDSelectorBitmap(std::shared_ptr<arrow::BooleanArray> array)
            : ::faiss::IDSelectorBitmap(array->length(), array->values()->data_as<uint8_t>())
            , array(std::move(array)) {};
};

struct IDSelectorCallback : ::faiss::IDSelector {
    const arrow::compute::Expression filter;
    const TablePtr table;
    const InternalTablePtr atable;
    arrow::compute::ExecContext *ctx;
    const arrow::Schema &schema;
    const arrow::ChunkedArrayVector &columns;
    const size_t num_columns;

    IDSelectorCallback(arrow::compute::Expression filter, TablePtr table)
            : table(table)
            , filter(std::move(filter))
            , atable(table->get_table())
            , schema(*atable->schema())
            , ctx(table->get_context()->get_exec_context())
            , columns(atable->columns())
            , num_columns(columns.size()) {};

    ~IDSelectorCallback() override {}

    bool is_member_alt(::faiss::idx_t id) const {
        // TODO: Could make this more efficient, but wait for actual performance numbers first.
        //PE("filter"); Multithreading issue?
        auto row    = atable->Slice(id, 1)->CombineChunksToBatch().ValueUnsafe();
        auto result = arrow::compute::ExecuteScalarExpression(
            filter, *atable->schema(), arrow::Datum(row), ctx);
        auto array  = std::static_pointer_cast<arrow::BooleanArray>(result->make_array());
        bool answer = array->Value(0);
        //PL("filter");
        return answer;
    };

    bool is_member(::faiss::idx_t id) const final {
        //PE("filter"); Multithreading issue?
        std::vector<arrow::Datum> row_values;
        row_values.reserve(columns.size());
        for (const auto &col : columns) {
            arrow::Result<std::shared_ptr<arrow::Scalar>> scalar_res = col->GetScalar(id);
            row_values.emplace_back(*scalar_res);
        }

        // Construct ExecBatch from scalars
        arrow::compute::ExecBatch batch{std::move(row_values), 1};

        // Execute expression lazily
        auto result = arrow::compute::ExecuteScalarExpression(filter, batch, ctx);
        bool bvalue = result.ValueUnsafe().scalar_as<arrow::BooleanScalar>().value;
        return bvalue;
        //PL("filter");
    };
};

struct IDSelectorOffset : ::faiss::IDSelector {
    const IDSelector *ids;
    const int offset;
    IDSelectorOffset(const IDSelector *ids, const int offset): ids(ids), offset(offset) {}
    bool is_member(::faiss::idx_t id) const final { return ids->is_member(id + offset); }
    virtual ~IDSelectorOffset() {}
};


}  // namespace maximus::faiss