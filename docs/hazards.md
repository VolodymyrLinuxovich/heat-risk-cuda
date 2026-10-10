# Hazard specification

This file is the single definition that all four implementations (`cpu`, `cpu_cpp`,
`gpu_torch`, `gpu_cuda`) follow. Tests check them against it.

## These are project simplifications, not ETCCDI indices

ETCCDI's TX90p uses calendar-day percentiles from a 5-day moving window over a base period, and
its tropical-night index TR counts days with Tmin > 20 °C. This project uses one percentile per
location over the whole baseline array and a 22 °C hot-night cutoff. The results are not
comparable with published ETCCDI values.

## Inputs

| Array | Shape | Units | Meaning |
|---|---|---|---|
| `tmax` | `[scenario, location, day]` | °C | daily maximum temperature |
| `tmin` | `[scenario, location, day]` | °C | daily minimum temperature |
| `rh` | `[scenario, location, day]` | % (0 to 100) | daily mean relative humidity |
| `baseline_tmax` | `[location, baseline_day]` | °C | separate baseline used only for percentiles |

All data arrays share one floating dtype (`float32` or `float64`).

**RH pairing (a documented simplification).** The heat index is computed from each day's Tmax and
that day's mean RH. Real afternoon RH at the time of Tmax is usually lower than the daily mean,
so this tends to overstate the heat index.

## Thresholds

Computed once by `heat_risk.thresholds.percentile_thresholds`:

```
np.nanpercentile(baseline_tmax, [90, 95], axis=1, method="linear")
```

The per-location results are cast to the data dtype in that function and passed to every
implementation. No implementation computes percentiles itself. A location whose baseline is all
NaN gets NaN thresholds, and no day counts for TX90/TX95 there.

## Hazards

All comparisons are `>=`.

| Hazard | Index | A day counts when |
|---|---|---|
| `tx90` | 0 | `tmax >= tx90_threshold[location]` |
| `tx95` | 1 | `tmax >= tx95_threshold[location]` |
| `hot_night` | 2 | `tmin >= 22.0` |
| `heat_index` | 3 | `heat_index_f(tmax, rh) >= 89.6` (89.6 °F = 32 °C) |

### Heat index (NWS)

Reference: https://www.wpc.ncep.noaa.gov/html/heatindex_equation.shtml

With `T` in °F (`T = tmax * 9/5 + 32`) and `RH` in percent:

1. Simple (Steadman) estimate: `HI = 0.5 * (T + 61.0 + (T - 68.0) * 1.2 + RH * 0.094)`.
2. If `(HI + T) / 2 < 80`, the simple estimate is the result.
3. Otherwise use the Rothfusz regression:
   ```
   HI = -42.379 + 2.04901523*T + 10.14333127*RH - 0.22475541*T*RH
        - 0.00683783*T*T - 0.05481717*RH*RH + 0.00122874*T*T*RH
        + 0.00085282*T*RH*RH - 0.00000199*T*T*RH*RH
   ```
   - If `RH < 13` and `80 <= T <= 112`, subtract `((13 - RH) / 4) * sqrt((17 - |T - 95|) / 17)`.
   - If `RH > 85` and `80 <= T <= 87`, add `((RH - 85) / 10) * ((87 - T) / 5)`.

The result stays in °F and is compared with 89.6, so every implementation compares the same
quantity. The operations are evaluated in the same order in all four implementations.

## NaN policy

A NaN in any input a hazard needs (the threshold included) means that day does not count for
that hazard, and it breaks that hazard's consecutive run.

## Outputs

For every `(scenario, location)` and every hazard, in the hazard order above:

- `counts[scenario, location, hazard]`: number of days that count, in `[0, days]`.
- `longest_run[scenario, location, hazard]`: longest streak of consecutive counted days,
  in `[0, counts]`. It is 0 when no day counts and when `days == 0`.

Both are `int32`.
