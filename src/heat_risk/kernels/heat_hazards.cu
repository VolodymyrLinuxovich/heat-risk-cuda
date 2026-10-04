// heat_hazards: one thread per (scenario, location) row, looping over days. Two variants:
// heat_hazards_{f32,f64} read day-major inputs; heat_hazards_tiled_{f32,f64} read the original
// row-major layout through shared-memory tiles (see heat_hazards_tiled_body below).
//
// Day-major inputs: value(day, row) = x[day * n_rows + row], with
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

// Per-row counters, kept in registers. Comparisons involving NaN are false, which implements
// the NaN policy: such a day does not count and breaks the run.
struct RowState {
    int count[N_HAZARDS] = {0, 0, 0, 0};
    int cur[N_HAZARDS] = {0, 0, 0, 0};
    int best[N_HAZARDS] = {0, 0, 0, 0};

    __device__ void update(bool h0, bool h1, bool h2, bool h3) {
        bool hit[N_HAZARDS] = {h0, h1, h2, h3};
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

    __device__ void store(int* counts, int* runs, int row) const {
        #pragma unroll
        for (int h = 0; h < N_HAZARDS; ++h) {
            counts[row * N_HAZARDS + h] = count[h];
            runs[row * N_HAZARDS + h] = best[h];
        }
    }
};

// Day-major kernel: needs inputs transposed to [day][row] first.
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

    RowState st;
    for (int d = 0; d < n_days; ++d) {
        long long i = (long long)d * n_rows + row;
        T tx = tmax[i];
        st.update(tx >= thr90, tx >= thr95, tmin[i] >= hot_night_c,
                  heat_index_f<T>(tx, rh[i]) >= heat_index_cut_f);
    }
    st.store(counts, runs, row);
}

// Row-major tiled kernel: reads the original [scenario, location, day] layout, so no transpose is
// needed. A block of TILE_ROWS threads owns TILE_ROWS consecutive rows. For each chunk of
// TILE_DAYS days, the block copies the TILE_ROWS x TILE_DAYS tile of each input into shared
// memory with consecutive threads reading consecutive days of a row (coalesced), then each
// thread walks its own row in the tile. The +1 padding avoids shared-memory bank conflicts
// when threads read down their rows.
#define TILE_ROWS 128

template <typename T>
struct TileDays { static const int value = 64 / sizeof(T); };  // 16 floats or 8 doubles

template <typename T>
__device__ void heat_hazards_tiled_body(
    const T* __restrict__ tmax,   // [row][day]
    const T* __restrict__ tmin,   // [row][day]
    const T* __restrict__ rh,     // [row][day]
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
    const int TD = TileDays<T>::value;
    __shared__ T s_tmax[TILE_ROWS][TileDays<T>::value + 1];
    __shared__ T s_tmin[TILE_ROWS][TileDays<T>::value + 1];
    __shared__ T s_rh[TILE_ROWS][TileDays<T>::value + 1];

    const int row0 = blockIdx.x * TILE_ROWS;
    const int row = row0 + threadIdx.x;
    const bool active = row < n_rows;
    T thr90 = T(0), thr95 = T(0);
    if (active) {
        int loc = row % n_locations;
        thr90 = tx90[loc];
        thr95 = tx95[loc];
    }
    RowState st;

    for (int d0 = 0; d0 < n_days; d0 += TD) {
        const int width = min(TD, n_days - d0);
        __syncthreads();  // previous tile fully consumed
        for (int k = threadIdx.x; k < TILE_ROWS * TD; k += blockDim.x) {
            int r = k / TD;
            int dd = k - r * TD;
            int g = row0 + r;
            if (g < n_rows && dd < width) {
                long long i = (long long)g * n_days + d0 + dd;
                s_tmax[r][dd] = tmax[i];
                s_tmin[r][dd] = tmin[i];
                s_rh[r][dd] = rh[i];
            }
        }
        __syncthreads();  // tile loaded
        if (active) {
            for (int dd = 0; dd < width; ++dd) {
                T tx = s_tmax[threadIdx.x][dd];
                st.update(tx >= thr90, tx >= thr95, s_tmin[threadIdx.x][dd] >= hot_night_c,
                          heat_index_f<T>(tx, s_rh[threadIdx.x][dd]) >= heat_index_cut_f);
            }
        }
    }
    if (active) st.store(counts, runs, row);
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

#define HEAT_HAZARDS_TILED_ENTRY(NAME, T)                                              \
    extern "C" __global__ void __launch_bounds__(TILE_ROWS) NAME(                       \
        const T* __restrict__ tmax, const T* __restrict__ tmin, const T* __restrict__ rh, \
        const T* __restrict__ tx90, const T* __restrict__ tx95,                         \
        int* __restrict__ counts, int* __restrict__ runs,                               \
        int n_rows, int n_locations, int n_days, T hot_night_c, T heat_index_cut_f)     \
    {                                                                                   \
        heat_hazards_tiled_body<T>(tmax, tmin, rh, tx90, tx95, counts, runs,            \
                                   n_rows, n_locations, n_days, hot_night_c,            \
                                   heat_index_cut_f);                                   \
    }

HEAT_HAZARDS_TILED_ENTRY(heat_hazards_tiled_f32, float)
HEAT_HAZARDS_TILED_ENTRY(heat_hazards_tiled_f64, double)
