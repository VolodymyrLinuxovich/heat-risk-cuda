// heat_hazards: one thread per (scenario, location) row, looping over days.
//
// Inputs are day-major: value(day, row) = x[day * n_rows + row], with
// row = scenario * n_locations + location. Adjacent threads therefore read adjacent
// addresses on every day (coalesced loads). Outputs are row-major [row][hazard], matching
// the [scenario, location, hazard] layout of the other implementations.
//
// The heat index mirrors src/heat_risk/heat_index.py operation by operation. Compile with
// fma disabled (--fmad=false) and without fast math so results match NumPy.

#define N_HAZARDS 4

template <typename T>
__device__ T heat_index_f(T tmax_c, T rh) {
    T t = tmax_c * T(1.8) + T(32.0);

    T simple = T(0.5) * (t + T(61.0) + (t - T(68.0)) * T(1.2) + rh * T(0.094));

    T full = T(-42.379)
           + T(2.04901523) * t
           + T(10.14333127) * rh
           - T(0.22475541) * t * rh
           - T(0.00683783) * t * t
           - T(0.05481717) * rh * rh
           + T(0.00122874) * t * t * rh
           + T(0.00085282) * t * rh * rh
           - T(0.00000199) * t * t * rh * rh;

    if ((rh < T(13.0)) && (t >= T(80.0)) && (t <= T(112.0))) {
        T low_arg = (T(17.0) - fabs(t - T(95.0))) / T(17.0);
        T low_adj = ((T(13.0) - rh) / T(4.0)) * sqrt(low_arg);
        full = full - low_adj;
    }
    if ((rh > T(85.0)) && (t >= T(80.0)) && (t <= T(87.0))) {
        T high_adj = ((rh - T(85.0)) / T(10.0)) * ((T(87.0) - t) / T(5.0));
        full = full + high_adj;
    }
    return ((simple + t) / T(2.0) >= T(80.0)) ? full : simple;
}

template <typename T>
__device__ void heat_hazards_body(
    const T* __restrict__ tmax,   // [day][row]
    const T* __restrict__ tmin,   // [day][row]
    const T* __restrict__ rh,     // [day][row]
    const T* __restrict__ tx90,   // [location]
    const T* __restrict__ tx95,   // [location]
    int* __restrict__ counts,     // [row][hazard]
    int* __restrict__ runs,       // [row][hazard]
    int n_rows,
    int n_locations,
    int n_days,
    T hot_night_c,
    T heat_index_cut_f)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;
    int loc = row % n_locations;
    T thr90 = tx90[loc];
    T thr95 = tx95[loc];

    int count[N_HAZARDS] = {0, 0, 0, 0};
    int cur[N_HAZARDS] = {0, 0, 0, 0};
    int best[N_HAZARDS] = {0, 0, 0, 0};

    for (int d = 0; d < n_days; ++d) {
        long long i = (long long)d * n_rows + row;
        T tx = tmax[i];
        bool hit[N_HAZARDS];
        // Comparisons involving NaN are false, which implements the NaN policy.
        hit[0] = tx >= thr90;
        hit[1] = tx >= thr95;
        hit[2] = tmin[i] >= hot_night_c;
        hit[3] = heat_index_f<T>(tx, rh[i]) >= heat_index_cut_f;
        #pragma unroll
        for (int h = 0; h < N_HAZARDS; ++h) {
            if (hit[h]) {
                count[h] += 1;
                cur[h] += 1;
                if (cur[h] > best[h]) best[h] = cur[h];
            } else {
                cur[h] = 0;
            }
        }
    }
    #pragma unroll
    for (int h = 0; h < N_HAZARDS; ++h) {
        counts[row * N_HAZARDS + h] = count[h];
        runs[row * N_HAZARDS + h] = best[h];
    }
}

#define HEAT_HAZARDS_ENTRY(NAME, T)                                                    \
    extern "C" __global__ void NAME(                                                    \
        const T* __restrict__ tmax, const T* __restrict__ tmin, const T* __restrict__ rh, \
        const T* __restrict__ tx90, const T* __restrict__ tx95,                         \
        int* __restrict__ counts, int* __restrict__ runs,                               \
        int n_rows, int n_locations, int n_days, T hot_night_c, T heat_index_cut_f)     \
    {                                                                                   \
        heat_hazards_body<T>(tmax, tmin, rh, tx90, tx95, counts, runs,                  \
                             n_rows, n_locations, n_days, hot_night_c, heat_index_cut_f); \
    }

// Plain C names so a cached cubin can be loaded without NVRTC's template name mapping.
HEAT_HAZARDS_ENTRY(heat_hazards_f32, float)
HEAT_HAZARDS_ENTRY(heat_hazards_f64, double)
