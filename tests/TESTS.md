# Test Suite — AutoDNA

## test_packet.py

Tests the `Packet` dataclass (`stm32/bin_parser/packet.py`).

Checks sensor name resolution from chunk IDs (gyro=0x01, accel=0x02, mag=0x03), unknown ID
fallback format, `sample_count` property, and `__repr__` content. No I/O — pure unit tests.

---

## test_stm_utils.py

Tests the binary parser (`stm32/bin_parser/stm_utils.py`).

**CRC16** — correctness, incremental update matches bulk compute, byte-order sensitivity.

**Byte unstuffing** — `0xFE` escape sequences produce correct values, roundtrip with stuffing,
incomplete escape raises `ValueError`.

**Packet parsing** — valid single/multi-chunk packets, bad header returns `None`, bad CRC returns
`None`, extreme int16 values and max timestamp parse correctly.

**File parsing** — synthetic `.BIN` streams built in-test using the same packet format as the
STM32 firmware; verifies packet count, sensor set, and timestamps without hardware.

**NPZ roundtrip** — `save_to_npz` + `load_from_npz` round-trip preserves sensor names, total
sample count, and raw data values.

---

## test_preprocessing.py

Tests the signal processing pipeline (`ai/preprocessing.py`).

**`smooth_signal`** — flat signal unchanged, window=1 is identity, known output values for a
spike input, spike amplitude is reduced.

**`lowpass_filter`** — raises `ValueError` for non-positive `fs`, non-positive `cutoff`, or
cutoff ≥ Nyquist. Correctly attenuates a 20 Hz component from a mixed 1 Hz + 20 Hz signal
while preserving the 1 Hz component. Preserves DC.

**`resample_sensors_to_common_grid`** — all sensors share identical timestamps after resampling,
output timestamps start at zero, millisecond timestamps are correctly detected and converted to
seconds, flat signal stays flat through interpolation.

**`preprocess_sensor_data`** — axes are **not** independently normalised; a 10×/1×/0.1×
amplitude ratio across axes is preserved after processing. This was explicitly fixed after a past
regression and the test guards against it being re-introduced. Timestamps pass through unchanged.

**Real data** (uses committed `data/training_data/parsed_data/LOG001.npz` — runs in CI):

| Test                                               | What it checks                                    |
| -------------------------------------------------- | ------------------------------------------------- |
| `test_real_npz_resamples_to_50hz`                  | Median timestamp step is 0.020 s after resampling |
| `test_real_npz_duration_retained_after_resampling` | Less than 2 % duration lost                       |
| `test_real_npz_all_finite_after_preprocessing`     | No NaN or inf in any axis                         |
| `test_real_npz_gyro_has_nonzero_variance`          | Gyro Z std > 0 — signal not destroyed             |

---

## test_features.py

Tests feature extraction (`ai/features.py`).

**Shape and dtype** — output is `(N_FEATURES,)` float32.

**Specific computed values** — constant gyro_z input: max, min, mean match the constant,
std = 0. Positive/negative fraction features return 1.0 when all samples are above/below
threshold. Zero gyro_z produces zero fractions.

**NaN safety** — when gyro_z std is zero, `np.corrcoef` would return NaN; the explicit `0.0`
branch is tested directly. All-zeros window produces no NaN.

**Real data** (uses `LOG001.npz` + `LOG001_labels.json` — runs in CI):

| Test                                           | What it checks                                                  |
| ---------------------------------------------- | --------------------------------------------------------------- |
| `test_feature_vectors_all_finite_on_real_data` | Every feature in every window is finite                         |
| `test_gyro_z_std_higher_in_turn_windows`       | Turn windows have >1.2× higher gyro_z std than non-turn windows |

The second test verifies the core assumption the XGBoost turn classifier depends on. If
preprocessing ever corrupts, inverts, or suppresses the gyro_z signal, this fails before any
degraded model reaches inference.

---

## test_pipeline.py

Tests the label assignment logic and the full data pipeline end-to-end.

**`_label_window` boundary tests** (no I/O, pure logic):

| Test                      | Scenario                                           |
| ------------------------- | -------------------------------------------------- |
| Full overlap              | Event covers entire window → label returned        |
| Exactly 30 % overlap      | At threshold (`>=`) → label returned               |
| 29 % overlap              | Just below threshold → `none` (0) returned         |
| No overlap                | Event is outside window → `none`                   |
| Multiple competing events | Event with highest overlap wins                    |
| Turn + hill independent   | Both labels assigned independently from same event |
| Hill-only event           | Turn stays `none`, hill is assigned                |

**Pipeline smoke tests** (use `LOG001.npz` — run in CI):

| Test                                        | What it checks                                                                              |
| ------------------------------------------- | ------------------------------------------------------------------------------------------- |
| `test_pipeline_output_shape_on_real_data`   | Full chain (load → resample → preprocess → window → extract) produces `(N, 36)` with no NaN |
| `test_pipeline_window_count_formula`        | Window count matches `(N_samples − 100) // 25 + 1`                                          |
| `test_label_window_on_real_labels_no_crash` | `_label_window` returns 0/1/2 for every real label entry without error                      |

---

## test_integration.py

Integration tests against real drive recordings from `C:\Users\marom\Desktop\Voznje\`.
**Skipped automatically in CI** when that path is absent.

**BIN parsing** — real LOG001.BIN has packets, all three sensors present, timestamps positive and
monotonically increasing, `data` shape is `(M, 3)` int16.

**BIN vs NPZ consistency** — sensor names and sample counts from live parsing match the
reference `.npz`.

**Label schema** — validates JSON schema version `autodna-v2-multilayer`, required fields
(`t_start`, `t_end`, `straight`), and valid enum values for `dir` and `intensity` on turn/hill
events.

**Label timeline** — every `t_end` falls within the recording duration from the NPZ.
