// heat_hazards.hpp: the C++ CPU implementation of docs/hazards.md, header only.
//
// The heat index mirrors src/heat_risk/heat_index.py and kernels/heat_hazards.cu operation by
// operation. Build without fast math and with floating-point contraction off
// (-ffp-contract=off) so no fused multiply-add changes float32 results versus NumPy.
//
// Layout: inputs are row-major [scenario, location, day], i.e. value(row, day) =
// x[row * n_days + day] with row = scenario * n_locations + location. Outputs are
// [row][hazard], hazards ordered tx90, tx95, hot_night, heat_index.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <thread>
#include <vector>

namespace heat_risk {

constexpr int kHazards = 4;

template <typename T>
T heat_index_f(T tmax_c, T rh) {
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
        T low_arg = (T(17.0) - std::fabs(t - T(95.0))) / T(17.0);
        T low_adj = ((T(13.0) - rh) / T(4.0)) * std::sqrt(low_arg);
        full = full - low_adj;
    }
    if ((rh > T(85.0)) && (t >= T(80.0)) && (t <= T(87.0))) {
        T high_adj = ((rh - T(85.0)) / T(10.0)) * ((T(87.0) - t) / T(5.0));
        full = full + high_adj;
    }
    return ((simple + t) / T(2.0) >= T(80.0)) ? full : simple;
}

// Running counters for one row. A comparison involving NaN is false, so a NaN day does not
// count and breaks the run (the NaN policy).
struct RowState {
    int32_t count[kHazards] = {0, 0, 0, 0};
    int32_t cur[kHazards] = {0, 0, 0, 0};
    int32_t best[kHazards] = {0, 0, 0, 0};

    void update(const bool (&hit)[kHazards]) {
        for (int h = 0; h < kHazards; ++h) {
            if (hit[h]) {
                count[h] += 1;
                cur[h] += 1;
                best[h] = std::max(best[h], cur[h]);
            } else {
                cur[h] = 0;
            }
        }
    }
};

template <typename T>
void evaluate_rows(const T* tmax, const T* tmin, const T* rh, const T* tx90, const T* tx95,
                   int32_t* counts, int32_t* runs, int64_t row_begin, int64_t row_end,
                   int64_t n_locations, int64_t n_days, T hot_night_c, T heat_index_cut_f) {
    for (int64_t row = row_begin; row < row_end; ++row) {
        const int64_t loc = row % n_locations;
        const T thr90 = tx90[loc];
        const T thr95 = tx95[loc];
        const int64_t base = row * n_days;
        RowState st;
        for (int64_t d = 0; d < n_days; ++d) {
            const T tx = tmax[base + d];
            const bool hit[kHazards] = {
                tx >= thr90,
                tx >= thr95,
                tmin[base + d] >= hot_night_c,
                heat_index_f<T>(tx, rh[base + d]) >= heat_index_cut_f,
            };
            st.update(hit);
        }
        for (int h = 0; h < kHazards; ++h) {
            counts[row * kHazards + h] = st.count[h];
            runs[row * kHazards + h] = st.best[h];
        }
    }
}

// Number of worker threads for n_rows > 0 rows. n_threads <= 0 means hardware_threads, which
// may be 0 when the count is unknown; then one thread is used. Always at least 1.
inline int64_t thread_count(int64_t n_rows, int64_t n_days, int n_threads,
                            unsigned hardware_threads) {
    if (n_threads <= 0) n_threads = static_cast<int>(hardware_threads);
    if (n_threads <= 0) n_threads = 1;
    // Below this many cells per thread, spawning costs more than it saves.
    constexpr int64_t kMinCellsPerThread = 1 << 16;
    const int64_t cells = n_rows * std::max<int64_t>(n_days, 1);
    const int64_t useful = std::max<int64_t>(1, cells / kMinCellsPerThread);
    return std::clamp<int64_t>(useful, 1, std::min<int64_t>(n_threads, n_rows));
}

// Splits rows into contiguous chunks, one per thread. n_threads <= 0 means
// std::thread::hardware_concurrency(). Rows are independent, so the result does not depend on
// the thread count.
template <typename T>
void evaluate(const T* tmax, const T* tmin, const T* rh, const T* tx90, const T* tx95,
              int32_t* counts, int32_t* runs, int64_t n_rows, int64_t n_locations,
              int64_t n_days, T hot_night_c, T heat_index_cut_f, int n_threads) {
    if (n_rows <= 0) return;
    const int64_t n =
        thread_count(n_rows, n_days, n_threads, std::thread::hardware_concurrency());
    if (n == 1) {
        evaluate_rows(tmax, tmin, rh, tx90, tx95, counts, runs, 0, n_rows, n_locations, n_days,
                      hot_night_c, heat_index_cut_f);
        return;
    }
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        const int64_t begin = n_rows * i / n;
        const int64_t end = n_rows * (i + 1) / n;
        workers.emplace_back([=] {
            evaluate_rows(tmax, tmin, rh, tx90, tx95, counts, runs, begin, end, n_locations,
                          n_days, hot_night_c, heat_index_cut_f);
        });
    }
    for (auto& w : workers) w.join();
}

}  // namespace heat_risk
