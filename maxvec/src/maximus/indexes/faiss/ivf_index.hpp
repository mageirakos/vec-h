#pragma once

#include <maximus/indexes/faiss/faiss_index.hpp>

namespace maximus::faiss {

class FaissIvfIndex : public FaissIndex {
public:
    FaissIvfIndex(): FaissIndex() {};
    ~FaissIvfIndex() = default;

    std::string to_string() = 0;

public:
protected:
};

}  // namespace maximus::faiss