#include <arrow/api.h>

#include <stdexcept>
#include <string>

#pragma once
namespace maximus {
enum ErrorCode {
    OK           = 0,
    ArrowError   = 1,
    MaximusError = 2,
};

class Status {
public:
    Status() = default;

    Status(int code, const std::string &msg) {
        this->_code    = code;
        this->_message = msg;
    }

    Status(const arrow::Status &status) {
        this->_code    = ErrorCode::ArrowError;
        this->_message = status.message();
    }

    explicit Status(int code) { this->_code = code; }

    explicit Status(ErrorCode code) { this->_code = code; }

    Status(ErrorCode code, const std::string &msg) {
        this->_code    = code;
        this->_message = msg;
    }

    int code() const { return _code; }

    bool ok() const { return _code == ErrorCode::OK; }

    static Status OK() { return maximus::Status(ErrorCode::OK); }

    const std::string &message() const { return _message; }

    std::string to_string() const {
        return "code: " + std::to_string(_code) + ", message: " + _message;
    }

private:
    int _code{};
    std::string _message{};
};

template<typename T>
void check_status(const T &expr) {
    if (!expr.ok()) {
        throw std::runtime_error("Maximus Error: " + std::to_string(static_cast<int>(expr.code())) +
                                 "; Message: " + expr.message() + "\n" + __FILE__ + ":" +
                                 std::to_string(__LINE__));
    }
}

template<typename T>
void check_status(const T &expr, const std::string &file, const std::string &line) {
    if (!expr.ok()) {
        throw std::runtime_error("Maximus Error: " + std::to_string(static_cast<int>(expr.code())) +
                                 "; Message: " + expr.message() + "\n" + file + ":" + line);
    }
}
}  // namespace maximus

#define CHECK_STATUS(expr) maximus::check_status(expr, __FILE__, std::to_string(__LINE__))