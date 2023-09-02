#pragma once


namespace fastnum {

enum class DeviceType {
    SSE,
    AVX2,
    NEON,
    CUDA,
    OCL
};

enum class OpType {
    SADD,
    SSUB,
    SMUL,
    SDIV,
    SSCALE,
    SMIN,
    SMAX,
    SLOG,
    SEXP,
};

enum class ElemType {
    FLOAT,
    DOUBLE
};

}