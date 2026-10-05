// C ABI over heat_hazards.hpp, loaded from Python with ctypes (see src/heat_risk/cpu_cpp.py).

#include "heat_hazards.hpp"

#define HEAT_HAZARDS_EXPORT(NAME, T)                                                         \
    extern "C" void NAME(const T* tmax, const T* tmin, const T* rh, const T* tx90,           \
                         const T* tx95, int32_t* counts, int32_t* runs, int64_t n_rows,      \
                         int64_t n_locations, int64_t n_days, T hot_night_c,                 \
                         T heat_index_cut_f, int n_threads) {                                \
        heat_risk::evaluate<T>(tmax, tmin, rh, tx90, tx95, counts, runs, n_rows,              \
                               n_locations, n_days, hot_night_c, heat_index_cut_f,           \
                               n_threads);                                                   \
    }

HEAT_HAZARDS_EXPORT(heat_hazards_cpp_f32, float)
HEAT_HAZARDS_EXPORT(heat_hazards_cpp_f64, double)

extern "C" int heat_hazards_cpp_abi_version() { return 1; }
