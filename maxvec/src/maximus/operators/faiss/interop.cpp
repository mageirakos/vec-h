#include <maximus/hashing/murmur_hash3.hpp>
#include <maximus/operators/faiss/interop.hpp>
#ifndef MAXIMUS_RELEASE_BUILD
#include <maximus/profiler/profiler.hpp>
#endif
::faiss::MetricType maximus::to_faiss_metric(const VectorDistanceMetric maximus_metric) {
    switch (maximus_metric) {
        case INNER_PRODUCT:
            return ::faiss::MetricType::METRIC_INNER_PRODUCT;
        case L2:
            return ::faiss::MetricType::METRIC_L2;
        default:
            throw std::invalid_argument("Unknown metric type");
    }
}

maximus::VectorDistanceMetric maximus::from_faiss_metric(const ::faiss::MetricType metric_type) {
    switch (metric_type) {
        case ::faiss::MetricType::METRIC_INNER_PRODUCT:
            return VectorDistanceMetric::INNER_PRODUCT;
        case ::faiss::MetricType::METRIC_L2:
            return VectorDistanceMetric::L2;
        default:
            throw std::invalid_argument("Unknown metric type");
    }
}

/************************************************************************************/
// Arrow Helper functions.                                                           /
/************************************************************************************/

float* maximus::raw_ptr_from_array(const arrow::FixedSizeListArray& array) {
    auto values    = std::static_pointer_cast<arrow::FloatArray>(array.values());
    int64_t offset = array.offset() * array.value_length();
    return (float*) values->raw_values() + offset;
}

namespace {

template <typename ListArrayT>
float* raw_ptr_from_list_array_impl(const ListArrayT& array) {
    if (array.length() == 0) {
        return nullptr;
    }

    // The 'values' child array contains all the float data concatenated together
    auto values = std::static_pointer_cast<arrow::FloatArray>(array.values());

    // Get the embedding dimension from the first element
    int64_t d = array.value_length(0);
    if (d == 0) {
        throw std::runtime_error(array.type()->ToString() + " has zero-length embeddings");
    }

    // Get the starting offset for the first element (accounting for array slicing)
    int64_t start_offset = array.value_offset(0);
#ifndef MAXIMUS_RELEASE_BUILD
    // Validate that all embeddings have the same dimension AND are stored contiguously
    PE("raw_ptr_from_array.val_cont");
    for (int64_t i = 0; i < array.length(); ++i) {
        if (array.value_length(i) != d) {
            throw std::runtime_error(array.type()->ToString() + " contains variable-length lists");
        }
        // Check that element i starts exactly where we expect for contiguous storage
        int64_t expected_offset = start_offset + i * d;
        int64_t actual_offset   = array.value_offset(i);
        if (actual_offset != expected_offset) {
            throw std::runtime_error(array.type()->ToString() + " elements are not contiguous in memory. "
                                     "Expected offset " +
                                     std::to_string(expected_offset) + " for element " +
                                     std::to_string(i) + ", but got " +
                                     std::to_string(actual_offset));
        }
    }
    PL("raw_ptr_from_array.val_cont");
#endif
    // Return a pointer to the start of the first element's data
    return const_cast<float*>(values->raw_values()) + start_offset;
}

}  // namespace

float* maximus::raw_ptr_from_array(const arrow::ListArray& array) {
    return raw_ptr_from_list_array_impl(array);
}

float* maximus::raw_ptr_from_array(const arrow::LargeListArray& array) {
    return raw_ptr_from_list_array_impl(array);
}

const float* maximus::get_embedding_raw_ptr(const std::shared_ptr<arrow::Array>& array) {
    // Check at runtime which way to extract the raw pointer (host side arrow pointers)
    if (array->type()->id() == arrow::Type::FIXED_SIZE_LIST) {
        return raw_ptr_from_array(*std::static_pointer_cast<arrow::FixedSizeListArray>(array));
    } else if (array->type()->id() == arrow::Type::LIST) {
        return raw_ptr_from_array(*std::static_pointer_cast<maximus::EmbeddingsArray>(array));
    } else if (array->type()->id() == arrow::Type::LARGE_LIST) {
        return raw_ptr_from_array(*std::static_pointer_cast<arrow::LargeListArray>(array));
    }
    throw std::runtime_error("Unsupported array type for embeddings: " + array->type()->ToString());
}

std::string maximus::metric_short_name(const ::faiss::MetricType metric) {
    switch (metric) {
        case ::faiss::MetricType::METRIC_INNER_PRODUCT:
            return "IP";
        case ::faiss::MetricType::METRIC_L2:
            return "L2";
        default:
            return "UNK";
    }
}

std::string maximus::compute_arrow_data_hash(const std::shared_ptr<arrow::Array>& vector_array,
                                    int dimension) {

    // Assumes a) FSL or LIST of Float32 embeddings b) single chunk (no slicing)
    const float* raw_data = nullptr;
    size_t total_floats   = 0;

    // 1. Extract Raw Pointer based on Array Type
    if (vector_array->type_id() == arrow::Type::FIXED_SIZE_LIST) {
        auto fsl_array = std::static_pointer_cast<arrow::FixedSizeListArray>(vector_array);
        raw_data       = raw_ptr_from_array(*fsl_array);
        //FSL: total floats = (number of rows * vector dimension)
        total_floats = (size_t)vector_array->length() * dimension; 
    } else if (vector_array->type_id() == arrow::Type::LIST) {
        auto list_array = std::static_pointer_cast<EmbeddingsArray>(vector_array);
        raw_data        = raw_ptr_from_array(*list_array);
        // LIST: total floats = length of the 'values' child array
        total_floats = list_array->values()->length(); 
    } else if (vector_array->type_id() == arrow::Type::LARGE_LIST) {
        auto large_list_array = std::static_pointer_cast<arrow::LargeListArray>(vector_array);
        raw_data              = raw_ptr_from_array(*large_list_array);
        // LARGE_LIST: total floats = length of the 'values' child array
        total_floats = large_list_array->values()->length();
    } else {
        throw std::runtime_error(
            "Unsupported Arrow Array type for hashing. Expected EmbeddingsArray, FixedSizeList, or LargeListArray.");
    }

    // 2. Compute Hash

    // Serialize inputs into a single buffer
    std::vector<uint8_t> buffer;

    // Hash Metadata (Length + Dimension)
    int64_t len = vector_array->length();
    buffer.insert(buffer.end(),
                  reinterpret_cast<uint8_t*>(&len),
                  reinterpret_cast<uint8_t*>(&len) + sizeof(len));

    buffer.insert(buffer.end(),
                  reinterpret_cast<uint8_t*>(&dimension),
                  reinterpret_cast<uint8_t*>(&dimension) + sizeof(dimension));

    // Hash selected Raw Data (First, Middle, Last vector) to avoid OOM
    if (raw_data && total_floats > 0) {
        int64_t num_vectors = len; // number of rows
        if (num_vectors > 0) {
            std::vector<int64_t> target_indices;
            target_indices.push_back(0);
            if (num_vectors > 2) {
                target_indices.push_back(num_vectors / 2);
            }
            if (num_vectors > 1) {
                target_indices.push_back(num_vectors - 1);
            }
            for (auto idx : target_indices) {
                const uint8_t* val_ptr = reinterpret_cast<const uint8_t*>(raw_data + (idx * dimension));
                buffer.insert(buffer.end(), val_ptr, val_ptr + (dimension * sizeof(float)));
            }
        }
    }

    // Compute MurmurHash
    uint32_t seed = 0;  // choose a fixed seed for determinism
    uint64_t hash_out[2];  // 128-bit output

    MurmurHash3_x64_128(buffer.data(),
                        static_cast<int>(buffer.size()),
                        seed,
                        hash_out);

    // Use lower 64 bits
    uint64_t hash = hash_out[0];

    // 3. Return Hex
    char hex[17];
    snprintf(hex, sizeof(hex), "%016llx",
             static_cast<unsigned long long>(hash));

    return std::string(hex);
}
