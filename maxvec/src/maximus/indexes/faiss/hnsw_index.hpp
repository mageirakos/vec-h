#pragma once

#include <maximus/indexes/faiss/faiss_index.hpp>

namespace maximus::faiss {

class FaissHNSWIndex : public FaissIndex {
public:
    FaissHNSWIndex(): FaissIndex() {};
    ~FaissHNSWIndex() = default;

    std::string to_string() = 0;

public:
protected:
};

}  // namespace maximus::faiss