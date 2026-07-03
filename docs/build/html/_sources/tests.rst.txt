Test Suite (``tests/``)
=======================

Unit tests are **network-free** and self-contained; the integration tests use
the committed real recordings and **auto-skip** when those files aren't present
(e.g. on a fresh CI checkout). Run everything with::

    pytest

CI (``.github/workflows/ci.yml``) runs the same suite with coverage on every push.

.. note::
   Test modules are not rendered with ``automodule`` — they rely on the path
   setup in ``tests/conftest.py`` and on pytest fixtures, so importing them
   outside a pytest run isn't meaningful. This page describes what each file
   covers instead.

``test_stm32.py`` — binary parser
---------------------------------
The STM32 ``.BIN`` decoder: CRC16 (incremental vs. bulk), byte un-stuffing and
its error path, ``parse_packet`` (rejects a bad CRC, decodes single and
multi-sensor packets), reading a whole synthetic ``.BIN`` stream, and the
``.npz`` save → load round-trip. Covers
:mod:`AutoDNA.stm32.bin_parser.packet` and
:mod:`AutoDNA.stm32.bin_parser.stm_utils`.

``test_ai.py`` — preprocessing, features, labels
------------------------------------------------
The IMU/ML chain (used elsewhere, not the live app):

* **preprocessing** — ``smooth_signal``; ``lowpass_filter`` (rejects bad params,
  attenuates 20 Hz, preserves DC); ``resample_sensors_to_common_grid`` (one
  shared 50 Hz grid, ms→s detection); the *axes are not individually
  normalised* invariant (a past-regression guard).
* **features** — ``extract_features`` output shape/dtype, known values on a
  constant window, and NaN-safety on zero-variance windows.
* **labels** — ``_label_window`` overlap threshold (≥30 %), highest-overlap
  wins, turn/hill independence.
* **real data** — feature vectors stay finite over a whole recording, and turn
  windows really do carry more gyro-Z variance than straights.

Covers :mod:`AutoDNA.ai.preprocessing`, :mod:`AutoDNA.ai.features`, and
``_label_window`` from :mod:`AutoDNA.app.ai_pipeline`.

``test_gps.py`` — the live GPS pipeline
---------------------------------------
Everything the app runs today:

* **turns / hills / distance** (:mod:`AutoDNA.app.gps_analysis`) — cumulative
  distance, turn detection (straight / left / right / one continuous segment),
  DEM-grade hill detection (up / down / flat, NaN-safe), elevation gain/loss.
* **elevation** (:mod:`AutoDNA.app.elevation`) — the on-disk cache round-trip
  and provider selection.
* **loader helpers** (:mod:`AutoDNA.app.data_loader`) — heading bearing, speed
  interpolation, fuel extraction (PID + fallback), per-point fuel rate.
* **path resolution** (:mod:`AutoDNA.app.paths`) — re-rooting a teammate's
  absolute drive path onto the local repo (the "click a stored drive" fix).
* **fuel records** (:mod:`AutoDNA.app.segment_records`) — bucket keys, first
  drive sets records with zero savings, a worse drive accrues savings and goes
  red, records are global.

``test_integration.py`` — real committed data (optional)
--------------------------------------------------------
Parses an actual ``LOG001.BIN``, checks sensors/timestamps/shape, verifies the
parsed data matches the reference ``.npz``, and validates the label-JSON schema.
Skips automatically if the data files aren't in the checkout.
