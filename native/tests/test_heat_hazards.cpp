// Native unit tests for src/heat_risk/native/heat_hazards.hpp. No test framework: each CHECK
// prints the failing line and the program exits non-zero. Run through CTest (see CMakeLists.txt).
// Cross-implementation parity against NumPy lives in tests/test_parity.py.

#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

#include "heat_hazards.hpp"

namespace {

int failures = 0;

#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::fprintf(stderr, "%s:%d: CHECK(%s)\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                   \
        }                                                                 \
    } while (0)

constexpr int TX90 = 0, TX95 = 1, HOT = 2, HI = 3;

double f_to_c(double f) { return (f - 32.0) / 1.8; }

template <typename T>
void test_nws_chart() {
    // (°F, RH %, heat index °F) read from the NWS chart, which rounds to whole °F.
    const double chart[][3] = {{80, 40, 80}, {90, 40, 91},  {100, 40, 109}, {90, 50, 95},
                               {90, 70, 106}, {86, 90, 105}, {80, 90, 86}};
    for (const auto& row : chart) {
        T hi = heat_risk::heat_index_f<T>(T(f_to_c(row[0])), T(row[1]));
        CHECK(std::fabs(double(hi) - row[2]) <= 1.0);
    }
    // Below 80 °F the simple formula is used: 0.5 * (70 + 61 + 2.4 + 4.7) = 69.05.
    CHECK(std::fabs(double(heat_risk::heat_index_f<T>(T(f_to_c(70.0)), T(50))) - 69.05) < 1e-4);
    // NaN propagates, so the comparison against the cutoff is false.
    const T nan = std::numeric_limits<T>::quiet_NaN();
    CHECK(std::isnan(heat_risk::heat_index_f<T>(nan, T(50))));
    CHECK(std::isnan(heat_risk::heat_index_f<T>(T(30), nan)));
}

template <typename T>
struct Result {
    std::vector<int32_t> counts, runs;
};

template <typename T>
Result<T> run(const std::vector<T>& tmax, const std::vector<T>& tmin, const std::vector<T>& rh,
              const std::vector<T>& tx90, const std::vector<T>& tx95, int64_t n_rows,
              int64_t n_loc, int64_t days, int threads) {
    Result<T> r{std::vector<int32_t>(n_rows * heat_risk::kHazards, -1),
                std::vector<int32_t>(n_rows * heat_risk::kHazards, -1)};
    heat_risk::evaluate<T>(tmax.data(), tmin.data(), rh.data(), tx90.data(), tx95.data(),
                           r.counts.data(), r.runs.data(), n_rows, n_loc, days, T(22.0),
                           T(89.6), threads);
    return r;
}

template <typename T>
void test_hot_night_runs_with_nan_break() {
    // One row, 9 days. Hot nights: Y Y N Y Y Y NaN Y Y -> count 7, longest run 3.
    const T nan = std::numeric_limits<T>::quiet_NaN();
    std::vector<T> tmin = {25, 25, 15, 25, 25, 25, nan, 25, 25};
    std::vector<T> tmax(9, T(10)), rh(9, T(40));
    std::vector<T> thr = {T(100)};
    auto r = run<T>(tmax, tmin, rh, thr, thr, 1, 1, 9, 1);
    CHECK(r.counts[HOT] == 7);
    CHECK(r.runs[HOT] == 3);
    CHECK(r.counts[TX90] == 0 && r.counts[TX95] == 0 && r.counts[HI] == 0);
}

template <typename T>
void test_thresholds_inclusive_and_per_location() {
    // Two locations, 4 days, thresholds differ per location; >= must count exact equality.
    std::vector<T> tmax = {30, 31, 31, 29,   // location 0
                           30, 31, 31, 29};  // location 1
    std::vector<T> tmin(8, T(22)), rh(8, T(10));
    std::vector<T> tx90 = {31, 30}, tx95 = {32, 31};
    auto r = run<T>(tmax, tmin, rh, tx90, tx95, 2, 2, 4, 1);
    CHECK(r.counts[0 * 4 + TX90] == 2 && r.runs[0 * 4 + TX90] == 2);
    CHECK(r.counts[0 * 4 + TX95] == 0);
    CHECK(r.counts[1 * 4 + TX90] == 3 && r.runs[1 * 4 + TX90] == 3);
    CHECK(r.counts[1 * 4 + TX95] == 2 && r.runs[1 * 4 + TX95] == 2);
    CHECK(r.counts[HOT] == 4 && r.runs[HOT] == 4);  // 22.0 is exactly the hot-night cutoff
}

template <typename T>
void test_zero_days_writes_zeros() {
    std::vector<T> empty, thr = {T(30), T(30)};
    auto r = run<T>(empty, empty, empty, thr, thr, 4, 2, 0, 3);
    for (int32_t v : r.counts) CHECK(v == 0);
    for (int32_t v : r.runs) CHECK(v == 0);
}

template <typename T>
void test_thread_count_does_not_change_results() {
    // 3 scenarios x 101 locations x 700 days of deterministic pseudo-random weather; large
    // enough that the multithreaded path splits the rows.
    const int64_t s = 3, n_loc = 101, days = 700, n_rows = s * n_loc;
    std::vector<T> tmax(n_rows * days), tmin(n_rows * days), rh(n_rows * days);
    uint64_t x = 88172645463325252ull;
    auto next = [&x] {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        return double(x >> 11) / double(1ull << 53);
    };
    for (size_t i = 0; i < tmax.size(); ++i) {
        tmax[i] = T(20 + 20 * next());
        tmin[i] = T(12 + 15 * next());
        rh[i] = T(100 * next());
    }
    std::vector<T> tx90(n_loc), tx95(n_loc);
    for (int64_t l = 0; l < n_loc; ++l) {
        tx90[l] = T(30 + 0.05 * l);
        tx95[l] = tx90[l] + T(2);
    }
    auto one = run<T>(tmax, tmin, rh, tx90, tx95, n_rows, n_loc, days, 1);
    for (int threads : {2, 3, 7, 0}) {
        auto many = run<T>(tmax, tmin, rh, tx90, tx95, n_rows, n_loc, days, threads);
        CHECK(many.counts == one.counts);
        CHECK(many.runs == one.runs);
    }
    for (size_t i = 0; i < one.counts.size(); ++i) {
        CHECK(one.counts[i] >= 0 && one.counts[i] <= days);
        CHECK(one.runs[i] >= 0 && one.runs[i] <= one.counts[i]);
    }
}

void test_thread_count_unknown_hardware() {
    // hardware_concurrency() may return 0. That must mean one thread, not zero (issue #4).
    const int64_t big = int64_t(1) << 24;
    CHECK(heat_risk::thread_count(1000, 365, 0, 0) == 1);
    CHECK(heat_risk::thread_count(big, 365, 0, 0) == 1);
    CHECK(heat_risk::thread_count(big, 365, -3, 0) == 1);
    CHECK(heat_risk::thread_count(big, 365, 0, 8) == 8);
    CHECK(heat_risk::thread_count(big, 365, 4, 0) == 4);
    CHECK(heat_risk::thread_count(3, 1 << 20, 0, 8) == 3);
    CHECK(heat_risk::thread_count(1, 1, 0, 8) == 1);
}

template <typename T>
void run_all() {
    test_nws_chart<T>();
    test_hot_night_runs_with_nan_break<T>();
    test_thresholds_inclusive_and_per_location<T>();
    test_zero_days_writes_zeros<T>();
    test_thread_count_does_not_change_results<T>();
}

}  // namespace

int main() {
    test_thread_count_unknown_hardware();
    run_all<float>();
    run_all<double>();
    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::puts("all native checks passed");
    return 0;
}
