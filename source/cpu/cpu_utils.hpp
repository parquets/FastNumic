#include <cstdlib>

namespace fastnum {
namespace cpu {

inline size_t alignSize(size_t sz, int n) {
    return (sz + n - 1) & -n;
}

inline void* alignedMalloc(size_t size, int alignment)
{
#if _MSC_VER
    return _aligned_malloc(size, alignment);
#else
    return std::aligned_alloc(alignment, size);
#endif
}

inline void alignedFree(void* mem)
{
#if _MSC_VER
    _aligned_free(mem);
#else
    return std::free(mem);
#endif
}


}
}