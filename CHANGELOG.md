# Changelog

All notable changes to Rtivity-Python are recorded here.
This project loosely follows [Semantic Versioning](https://semver.org/).

## [0.12.1] — 2026-11-01

### Fixed

- **Every channel was read from the wrong column.** `dam_parser._read_raw` took
  channel *k* from tab field `9 + k` (0-indexed), but in a 42-field DAM2 file
  0-indexed field 9 is the **light sensor** and channels are fields 10–41. So
  channel 1 carried the light column (0/1 per read), channel *k* carried tube
  *k − 1*, and tube 32 was never read. Every activity, sleep, rhythm and
  survival result is affected. Confirmed on three genuine rig exports: on a
  file whose light column varies, "channel 1" summed to exactly the number of
  lit reads. The docstrings described two incompatible layouts — a
  whitespace-split view (date as three fields, 44 in all) and a tab view with a
  non-existent trailing field — and the offset came from mixing them; both are
  rewritten to the tab layout. (`modules/data/dam_parser.py`)
- **The package reported the wrong version.** `pyproject.toml` still said
  0.11.0 under the v0.12.0 tag, so `pip show rtivity-python` reported 0.11.0.
  It now says 0.12.1.

### Added

- `tests/test_dam_parser_columns.py`: a synthetic DAM2 file built in the test,
  with distinct values in the light field, the field before it and every
  channel, so any one-column shift fails. Runs everywhere — unlike the existing
  parser tests, which need `~/Desktop/Rtivity_main` and skip without it (in CI,
  256 of 279 tests skip). All 35 fail on v0.12.0.

### Note on v0.12.0

v0.12.0 was tagged without a changelog entry. It contained the survival
default death window change (24 h → 12 h) and the session-scoped test fixture.

## [0.11.0] — 2026-06-25

Second pass after the first end-to-end run on real data: fixes a crash that
broke every plot, adds R-style per-metric Excel export, handles the
no-deaths case, and reorders the tabs. All 279 unit tests pass.

### Added

- **Per-metric Excel export matching the R Rtivity format.** Every analysis tab
  (Activity, Sleep, Rhythm, Survival) now has a **Download** button that writes a
  multi-sheet `.xlsx`: `Settings`, `Statistics` (N / mean / SEM / SD / median /
  quartiles per condition × level), `All data` (one row per animal, one column
  per level, with File/Labels/Channels/Start/End), and one sheet per level laid
  out condition-by-animal for direct paste into GraphPad Prism. Mirrors the R
  download (e.g. "Sleep ratio per light phase"). Survival exports a per-animal
  duration/event table plus median-survival, log-rank, and flagged-animal
  sheets. (`modules/data/export.py` `write_metric_workbook`; `app.py` `dl_*`
  handlers)

### Fixed

- **Every plot crashed with "Value function must be callable."** The render
  outputs `km_plot`, `actogram_plot`, and `daily_profile_plot` shared names with
  the imported Plotly helpers, so each ended up calling itself (a linter had
  also stripped the now-"unused" imports, compounding it). The Kaplan-Meier
  curve, actogram, and daily-profile plots were all affected. Fixed by aliasing
  the imports (`plot_km` / `plot_actogram` / `plot_daily_profile`). This only
  surfaced once data loading started working.
- **No feedback when no animal dies.** The Survival tab now shows "There is no
  death animal detected at this silence window" (and the median/log-rank tables
  say so too) instead of an empty Kaplan-Meier plot and blank median rows.

### Changed

- **Survival analysis moved to the second tab**, right after Data selection.

## [0.10.0] — 2026-06-25

This release overhauls the Data-selection tab's manual metadata setup, fixes
two bugs that made data loading silently fail, refines survival death-detection
to match the R original, and corrects stale claims in the README. All 270 unit
tests pass.

### Added

- **R-style manual metadata builder.** The "Manual Setup" mode on the
  Data-selection tab now mirrors the R app: pick a file, type a condition name,
  tick its channels in a checkbox grid (only as-yet-unassigned channels are
  offered, so channels can't be double-assigned), and click **Add condition**.
  Conditions accumulate in a table you can select rows from and remove, or clear
  entirely. Any number of conditions is supported — one (whole monitor) or many.
  This replaces the earlier fixed two-group layout *and* a short-lived
  "Number of groups" numeric field that could be set to 0 and crash the panel.
  (`app.py` `manual_dates_ui`, `manual_channels_ui`, `manual_conditions_table`,
  and the `manual_*` observers; `modules/data/metadata.py`
  `build_manual_conditions`, `parse_channel_spec`, `_normalize_channels`)
- **Per-file Start / Finish datetime fields in Manual Setup**, auto-filled from
  each DAM file's actual recorded time range and editable, so the analysis
  window can be set without uploading a metadata sheet.
  (`app.py` `_detect_time_range`, using `dam_parser.get_time_range`)
- **Flexible channel input.** `build_manual_conditions` accepts channels either
  as a spec string (`1-16`, `1,3,5-10`) or as a list of integers (what the
  checkbox grid produces), with validation: empty condition names, channels
  outside 1–32, the same channel assigned twice within a file, and start ≥
  finish all raise a clear error. (`modules/data/metadata.py`
  `parse_channel_spec`, `_normalize_channels`, `build_manual_conditions`)
- **"Save Metadata" download in Manual Setup** — exports the conditions you
  built as an `.xlsx`, equivalent to the R app's `saveMetadata` button.
  (`app.py` `save_metadata_btn`)
- **"Flagged animals — activity after inferred death" table** on the Survival
  tab. Lists any animal that recorded movement *after* its inferred time of
  death, with the inferred death day, total post-death activity counts, number
  of active bins after death, and the day of its last movement. These are the
  ambiguous/outlier cases (e.g. a monitor glitch, or a threshold set too short)
  that the death rule otherwise hides. (`modules/analysis/survival.py`
  `flag_ambiguous_deaths`, `app.py` `flagged_deaths_table`)
- New tests: `tests/test_survival_death_detection.py` (death-rule semantics and
  flagging) and manual-metadata-builder cases in `tests/test_metadata.py`.

### Changed

- **Survival death detection now matches the R original's intent.** An animal
  is called dead if **any** continuous silent run within the selected time
  window reaches the threshold (e.g. 12 h), and death is dated to the **first**
  such run — not the longest, and not merely whether the recording ends in
  silence. If no silent run reaches the threshold the animal is right-censored,
  even if it happens to be quiet at the very end. Any movement after the
  inferred death is treated as artifact (and surfaced in the new flagged-animals
  table rather than changing the death time). Silence is measured from the
  actual timestamps, so it does not assume 1-minute sampling the way the R
  code's "row count ÷ 60" does. (`modules/analysis/survival.py`
  `build_survival_data`, `_first_lethal_silence`)
- **README scope section corrected and expanded.** Removed the inaccurate claim
  that actogram visualizations are missing (they are implemented in
  `modules/ui/plots.py`), and moved IS/IV/RA out of "added vs the R version"
  (the R version computes them too, via the `ActCR` package). The "What's not
  here yet" list now accurately reflects the genuine gaps confirmed by a full
  read of the R source: DFA/fractal analysis, per-day breakdown of the circadian
  indices, the full (non-wrapped) chronogram / sleep chronogram /
  cumulative-activity plots, the bin-converter utility, the data-integrity
  imputation tab (including the note that this rewrite's parser silently zeros
  missing activity values rather than imputing them), and per-tab Excel export.

### Fixed

- **"Load data" now works in Manual Setup mode.** The button's handler only ever
  read an uploaded metadata `.xlsx`, so clicking it while in Manual Setup did
  nothing. It now branches on the selected metadata source and builds the
  dataset from the manual entries.
- **Restored the load-status indicator.** A previous edit had repurposed the
  function that rendered the "Loaded N animals across groups…" confirmation, so
  that panel rendered nothing. It is restored, and load failures (bad channel
  spec, overlapping channels, invalid dates, missing files) now show a clear
  message in the panel and a notification, instead of only printing to the
  console.
- **Activity "Day × Phase" grouping for bout metrics.** For the "Activity per
  bout" and "Bout duration" metrics, the Day × Phase view was calling the
  day-only summary functions and silently showing day-only results. It now uses
  the correct `bout_activity_by_day_phase` / `bout_duration_by_day_phase`.
- **Manual Setup "Number of groups" could be set to 0** and crash the panel with
  `int() argument must be a string … not 'NoneType'`. The numeric field is gone
  entirely in the new condition-builder UI, so the failure mode no longer
  exists.

### Notes for reviewers / users

- The death rule treats post-threshold-silence movement as artifact and keeps
  the early death time. This is a deliberate scientific convention (and what the
  R original does). The new flagged-animals table exists so these cases are
  visible; if your data shows many flagged animals, consider whether the silence
  threshold is too short.
- `.claude/` (local Claude Code settings and the dev-server launch config) is
  now git-ignored; it is not part of the application.
