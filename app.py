"""
Rtivity — Python edition
========================
Shiny for Python application for the analysis of Drosophila Activity Monitor
(DAM) data.  Replicates the R/Shiny Rtivity app.

Run with:
    shiny run app.py --reload
or:
    python app.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from shiny import App, reactive, render, req, ui

sys.path.insert(0, str(Path(__file__).parent))

from modules.data.dam_parser import get_time_range
from modules.data.data_store import DataStore, Settings
from modules.data.export import write_metric_workbook
from modules.data.metadata import build_manual_conditions
from modules.analysis.activity import (
    activity_by_day,
    activity_by_day_phase,
    activity_by_phase,
    bout_activity_by_day,
    bout_activity_by_day_phase,
    bout_activity_by_phase,
    bout_duration_by_day,
    bout_duration_by_day_phase,
    bout_duration_by_phase,
    detect_activity_bouts,
    summary_stats,
)
from modules.analysis.rhythm import (
    cosinor_summary,
    dominant_period_by_animal,
    is_iv_ra,
    periodogram_by_animal,
)
from modules.analysis.sleep import (
    annotate_sleep,
    detect_sleep_bouts,
    sleep_bout_count_by_day,
    sleep_bout_count_by_phase,
    sleep_bout_duration_by_day,
    sleep_bout_duration_by_phase,
    sleep_latency_by_day,
    sleep_latency_mean,
    sleep_ratio_by_day,
    sleep_ratio_by_day_phase,
    sleep_ratio_by_phase,
    tst_by_day,
    tst_mean,
    waso_by_day,
    waso_mean,
)
from modules.analysis.survival import (
    build_survival_data,
    flag_ambiguous_deaths,
    kaplan_meier_curves,
    logrank_test,
    median_survival,
)
from modules.ui.plots import (
    actogram_plot as plot_actogram,
    box_plot,
    daily_profile_plot as plot_daily_profile,
    km_plot as plot_km,
    periodogram_plot,
)

# ── helpers ───────────────────────────────────────────────────────────────────

def _fig_html(fig) -> ui.HTML:
    """Convert a plotly figure to Shiny HTML output."""
    return ui.HTML(fig.to_html(full_html=False, include_plotlyjs="cdn"))


def _sidebar_card(*args, title: str = "", width: int = 280) -> ui.Tag:
    return ui.sidebar(
        ui.h5(title, class_="fw-bold mb-3") if title else None,
        *args,
        width=width,
        bg="#f8f9fa",
    )


def _compress_channels(channels: list[int]) -> str:
    """Render a sorted channel list as compact ranges: [1,2,3,4,7] -> '1-4, 7'."""
    chans = sorted(set(channels))
    if not chans:
        return ""
    parts, run_start, prev = [], chans[0], chans[0]
    for c in chans[1:] + [None]:
        if c == prev + 1:
            prev = c
            continue
        parts.append(f"{run_start}-{prev}" if prev > run_start else f"{run_start}")
        run_start = prev = c
    return ", ".join(parts)


def _detect_time_range(path: str) -> tuple[str, str]:
    """Best-effort detection of a DAM file's [first, last] timestamp, used to
    pre-fill Start/Finish datetime fields in Manual Setup mode."""
    try:
        start, stop = get_time_range(path)
        return (str(start), str(stop))
    except Exception:
        return ("", "")


# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════

app_ui = ui.page_navbar(
    ui.nav_spacer(),

    # ── Tab 1 · Data selection ────────────────────────────────────────────────
    ui.nav_panel(
        "Data selection",
        ui.div(
            ui.h1("Rtivity", class_="text-center fw-bold mt-3",
                  style="font-size:60px;"),
            ui.h4(
                "Automated analysis of animal activity and sleep patterns",
                class_="text-center text-muted mb-1",
            ),
            ui.h6("Python Edition", class_="text-center text-muted mb-4"),
            class_="mb-2",
        ),
        ui.layout_sidebar(
            _sidebar_card(
                ui.input_file("dam_files", "DAM monitor files (.txt)",
                              multiple=True, accept=[".txt"]),
                ui.input_radio_buttons(
                "meta_mode", 
                "Metadata Source", 
                choices={"upload": "Upload .xlsx", "manual": "Manual Setup"}
            ),
            ui.panel_conditional(
                "input.meta_mode === 'upload'",
                ui.input_file("metadata_file", "Metadata file (.xlsx)", accept=[".xlsx"])
            ),
            ui.panel_conditional(
                "input.meta_mode === 'manual'",
                ui.output_ui("manual_dates_ui"),
                ui.hr(),
                ui.p("Build conditions one at a time: name it, tick its "
                     "channels, click Add. Repeat for each condition — "
                     "one is fine, or as many as you need.",
                     class_="text-muted small"),
                ui.input_select("manual_file", "Assign channels for file",
                                choices=[]),
                ui.input_text("manual_cond_name", "Condition name",
                              placeholder="e.g. Control"),
                ui.output_ui("manual_channels_ui"),
                ui.layout_columns(
                    ui.input_action_button("manual_selectall_btn",
                                           "Select all", class_="btn btn-sm btn-outline-secondary"),
                    ui.input_action_button("manual_clearsel_btn",
                                           "Clear ticks", class_="btn btn-sm btn-outline-secondary"),
                    col_widths=[6, 6],
                ),
                ui.input_action_button("manual_add_btn", "Add condition",
                                       class_="btn btn-primary w-100 mt-2 mb-2"),
                ui.output_data_frame("manual_conditions_table"),
                ui.layout_columns(
                    ui.input_action_button("manual_remove_btn", "Remove selected",
                                           class_="btn btn-sm btn-outline-danger"),
                    ui.input_action_button("manual_clear_btn", "Clear all",
                                           class_="btn btn-sm btn-outline-danger"),
                    col_widths=[6, 6],
                ),
                ui.download_button("save_metadata_btn", "Save Metadata",
                                   class_="btn btn-outline-secondary w-100 mt-2"),
            ),
                ui.hr(),
                ui.h6("Settings", class_="fw-bold"),
                ui.input_text("zt0", "Lights-on time (ZT0)", value="06:00:00"),
                ui.input_numeric("ld_period", "LD cycle period (h)", 24.0,
                                 min=1, max=48, step=0.5),
                ui.input_numeric("light_hours", "Light phase duration (h)", 12.0,
                                 min=1, max=24, step=0.5),
                ui.input_numeric("sleep_threshold_setting",
                                 "Sleep threshold (min inactivity)", 5,
                                 min=1, max=60, step=1),
                ui.input_numeric("bout_threshold_setting",
                                 "Activity bout gap threshold (min)", 5,
                                 min=1, max=60, step=1),
                ui.hr(),
                ui.input_action_button("load_btn", "Load data",
                                       class_="btn btn-success w-100"),
                title="Load data",
            ),
            ui.card(
                ui.output_ui("load_status"),
                ui.output_data_frame("animals_table"),
                full_screen=False,
            ),
        ),
    ),

    # ── Tab 2 · Survival analysis ─────────────────────────────────────────────
    ui.nav_panel(
        "Survival analysis",
        ui.layout_sidebar(
            _sidebar_card(
                ui.input_numeric("dead_window_h",
                                 "Silence window to call death (h)", 24,
                                 min=1, max=72, step=1),
                ui.input_action_button("surv_btn", "Run survival analysis",
                                       class_="btn btn-primary w-100"),
                title="Controls",
            ),
            ui.card(
                ui.h5("Kaplan-Meier curves"),
                ui.output_ui("km_plot"),
                ui.hr(),
                ui.h6("Median survival"),
                ui.output_data_frame("median_surv_table"),
                ui.hr(),
                ui.h6("Log-rank tests (pairwise)"),
                ui.output_data_frame("logrank_table"),
                ui.hr(),
                ui.h6("Flagged animals — activity after inferred death"),
                ui.p("An animal is called dead at the first silent stretch that "
                     "reaches the threshold above; any later movement is treated "
                     "as artifact and does not change the death time. The animals "
                     "below moved again after that point — review them in case a "
                     "monitor glitch or a too-short threshold mislabeled a living "
                     "animal. An empty table means no such outliers.",
                     class_="text-muted small"),
                ui.output_data_frame("flagged_deaths_table"),
                ui.download_button("dl_survival", "Download survival (.xlsx)",
                                   class_="btn btn-outline-secondary btn-sm mt-2"),
            ),
        ),
    ),

    # ── Tab 3 · Visual inspection ─────────────────────────────────────────────
    ui.nav_panel(
        "Visual inspection",
        ui.layout_sidebar(
            _sidebar_card(
                ui.input_select("vis_animal", "Animal", choices=[]),
                ui.input_numeric("vis_bin_min", "Bin size (min)", 30,
                                 min=1, max=120, step=1),
                ui.hr(),
                ui.input_select("vis_group", "Group for daily profile",
                                choices=[]),
                ui.input_numeric("vis_profile_bin",
                                 "Profile bin size (min)", 30,
                                 min=1, max=120, step=1),
                title="Controls",
            ),
            ui.card(
                ui.h5("Actogram"),
                ui.output_ui("actogram_plot"),
                ui.hr(),
                ui.h5("Average Daily Activity Profile"),
                ui.output_ui("daily_profile_plot"),
            ),
        ),
    ),

    # ── Tab 4 · Activity analysis ─────────────────────────────────────────────
    ui.nav_panel(
        "Activity analysis",
        ui.layout_sidebar(
            _sidebar_card(
                ui.input_slider("act_bout_window",
                                "Bout gap threshold (min)", 0.5, 60, 5, step=0.5),
                ui.input_action_button("act_analyse_btn",
                                       "Run bout detection",
                                       class_="btn btn-primary w-100 mb-3"),
                ui.hr(),
                ui.input_select("act_groupby", "Group by",
                                {"phase": "Light / Dark",
                                 "day": "Experiment day",
                                 "day_phase": "Day × Phase"}),
                ui.input_select("act_metric", "Metric",
                                {"total": "Total activity",
                                 "bout_activity": "Activity per bout",
                                 "bout_duration": "Bout duration (s)"}),
                ui.input_action_button("act_plot_btn", "Update plot",
                                       class_="btn btn-dark w-100"),
                title="Controls",
            ),
            ui.card(
                ui.output_ui("activity_plot"),
                ui.hr(),
                ui.h6("Descriptive statistics"),
                ui.output_data_frame("activity_stats"),
                ui.download_button("dl_activity", "Download metric (.xlsx)",
                                   class_="btn btn-outline-secondary btn-sm mt-2"),
            ),
        ),
    ),

    # ── Tab 5 · Sleep analysis ────────────────────────────────────────────────
    ui.nav_panel(
        "Sleep analysis",
        ui.layout_sidebar(
            _sidebar_card(
                ui.input_slider("sleep_threshold", "Sleep threshold (min)", 1, 60, 5),
                ui.input_action_button("sleep_analyse_btn",
                                       "Run sleep annotation",
                                       class_="btn btn-primary w-100 mb-3"),
                ui.hr(),
                ui.input_select("sleep_groupby", "Group by",
                                {"phase": "Light / Dark",
                                 "day": "Experiment day",
                                 "day_phase": "Day × Phase"}),
                ui.input_select("sleep_metric", "Metric",
                                {"ratio": "Sleep ratio",
                                 "tst": "Total sleep time (h)",
                                 "waso": "Wake after sleep onset (h)",
                                 "latency": "Sleep latency (min)",
                                 "bout_duration": "Sleep bout duration (min)",
                                 "bout_count": "Sleep bout count"}),
                ui.input_action_button("sleep_plot_btn", "Update plot",
                                       class_="btn btn-dark w-100"),
                title="Controls",
            ),
            ui.card(
                ui.output_ui("sleep_plot"),
                ui.hr(),
                ui.h6("Descriptive statistics"),
                ui.output_data_frame("sleep_stats"),
                ui.download_button("dl_sleep", "Download metric (.xlsx)",
                                   class_="btn btn-outline-secondary btn-sm mt-2"),
            ),
        ),
    ),

    # ── Tab 6 · Rhythm analysis ───────────────────────────────────────────────
    ui.nav_panel(
        "Rhythm analysis",
        ui.layout_sidebar(
            _sidebar_card(
                ui.input_select("rhythm_method", "Periodogram method",
                                {"chi_sq": "Chi-square (Sokolove & Bushell)",
                                 "lomb_scargle": "Lomb-Scargle"}),
                ui.input_numeric("rhythm_period_from", "Period range — from (h)",
                                 16.0, min=1, max=48),
                ui.input_numeric("rhythm_period_to", "Period range — to (h)",
                                 32.0, min=1, max=96),
                ui.input_numeric("rhythm_period_step", "Step (h)",
                                 0.5, min=0.1, max=2, step=0.1),
                ui.input_action_button("rhythm_btn", "Run analysis",
                                       class_="btn btn-primary w-100 mb-3"),
                ui.hr(),
                ui.input_select("rhythm_view", "View",
                                {"periodogram": "Periodogram",
                                 "dominant_period": "Dominant period",
                                 "cosinor": "Cosinor fit",
                                 "is_iv_ra": "IS / IV / RA"}),
                title="Controls",
            ),
            ui.card(
                ui.output_ui("rhythm_plot"),
                ui.hr(),
                ui.output_data_frame("rhythm_table"),
                ui.download_button("dl_rhythm", "Download metric (.xlsx)",
                                   class_="btn btn-outline-secondary btn-sm mt-2"),
            ),
        ),
    ),

    title=ui.tags.span("Rtivity", style="font-weight:bold; font-size:1.2em;"),
    id="main_nav",
    navbar_options=ui.navbar_options(bg="#2c3e50"),
    fluid=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# Server
# ══════════════════════════════════════════════════════════════════════════════

def server(input, output, session):

    # ── core reactive state ───────────────────────────────────────────────────

    r_store      = reactive.value(None)   # DataStore
    r_load_error = reactive.value(None)   # last load failure message, if any
    r_conditions = reactive.value([])     # manual-mode conditions accumulator:
                                          #   [{"file", "label", "channels": (int,)}]
    r_ab_df   = reactive.value(None)   # data with activity bouts
    r_sleep   = reactive.value(None)   # data with sleep annotation + bouts
    r_pgram   = reactive.value(None)   # periodogram DataFrame
    r_cosinor = reactive.value(None)   # cosinor DataFrame
    r_isivra  = reactive.value(None)   # IS/IV/RA DataFrame
    r_surv_df = reactive.value(None)   # survival data
    r_km_df   = reactive.value(None)   # KM curves
    r_flagged = reactive.value(None)   # animals active after inferred death

    # ── manual metadata helpers ───────────────────────────────────────────────

    def _collect_manual_entries() -> list[dict]:
        """Reshape the accumulated r_conditions list + per-file date inputs into
        the file-grouped structure build_manual_conditions() expects."""
        dam_files = input.dam_files() or []
        idx_of = {f["name"]: i for i, f in enumerate(dam_files)}

        by_file: dict[str, list[dict]] = {}
        for cond in r_conditions():
            by_file.setdefault(cond["file"], []).append(
                {"label": cond["label"], "channels": list(cond["channels"])}
            )

        entries = []
        for fname, groups in by_file.items():
            i = idx_of.get(fname)
            sid, eid = f"start_dt_{i}", f"stop_dt_{i}"
            entries.append({
                "file": fname,
                "start_datetime": input[sid]() if i is not None and sid in input else "",
                "stop_datetime":  input[eid]() if i is not None and eid in input else "",
                "groups": groups,
            })
        return entries

    # ── data loading ──────────────────────────────────────────────────────────

    @reactive.effect
    @reactive.event(input.load_btn)
    def _load():
        dam_files = input.dam_files()
        mode = input.meta_mode()

        if not dam_files:
            r_store.set(None)
            r_load_error.set("Upload DAM monitor files (.txt) first.")
            return

        if mode != "manual" and not input.metadata_file():
            r_store.set(None)
            r_load_error.set(
                "Upload a metadata .xlsx file, or switch to Manual Setup."
            )
            return

        try:
            tmpdir = Path(tempfile.mkdtemp(prefix="rtivity_"))
            for f in dam_files:
                shutil.copy(f["datapath"], tmpdir / f["name"])

            settings = Settings(
                zt0=input.zt0(),
                ld_period=float(input.ld_period()),
                light_hours=float(input.light_hours()),
                sleep_threshold_min=int(input.sleep_threshold_setting()),
                bout_threshold_min=int(input.bout_threshold_setting()),
            )
            store = DataStore(dam_dir=tmpdir, settings=settings)

            if mode == "manual":
                entries = _collect_manual_entries()
                conditions = build_manual_conditions(entries)
                store.load_from_dataframe(conditions, dam_dir=tmpdir)
            else:
                meta_path = tmpdir / "metadata.xlsx"
                shutil.copy(input.metadata_file()[0]["datapath"], meta_path)
                store.load_from_metadata(meta_path)

            # Invalidate all derived caches
            r_ab_df.set(None)
            r_sleep.set(None)
            r_pgram.set(None)
            r_cosinor.set(None)
            r_isivra.set(None)
            r_surv_df.set(None)
            r_km_df.set(None)
            r_flagged.set(None)

            r_load_error.set(None)
            r_store.set(store)

            # Populate animal / group selectors
            animals = store.animal_ids
            labels  = store.labels
            ui.update_select("vis_animal",  choices=animals,
                             selected=animals[0] if animals else None)
            ui.update_select("vis_group",   choices=labels,
                             selected=labels[0] if labels else None)

        except Exception as exc:
            r_store.set(None)
            r_load_error.set(str(exc))
            ui.notification_show(f"Load failed: {exc}", type="error", duration=8)
            print(traceback.format_exc())

    # ── derived data accessors ────────────────────────────────────────────────

    @reactive.calc
    def _data():
        req(r_store() is not None)
        return r_store().data

    # ── load status + animals table ───────────────────────────────────────────

    @output
    @render.ui
    def manual_dates_ui():
        """Per-file Start/Finish datetime fields, auto-filled from each file's
        recorded range. Depends only on the uploaded file set, so user edits
        persist (the block is not re-rendered while typing)."""
        files = input.dam_files()
        if not files:
            return ui.p("Please upload DAM monitor files (.txt) first.",
                        style="color: red;")

        cards = []
        for i, f in enumerate(files):
            start_val, stop_val = _detect_time_range(f["datapath"])
            cards.append(ui.div(
                ui.tags.small(f["name"], class_="fw-bold"),
                ui.layout_columns(
                    ui.input_text(f"start_dt_{i}", "Start", value=start_val),
                    ui.input_text(f"stop_dt_{i}", "Finish", value=stop_val),
                    col_widths=[6, 6],
                ),
                class_="mb-2",
            ))
        return ui.TagList(*cards)

    def _assigned_channels(filename: str) -> set[int]:
        """Channels already claimed by some condition on the given file."""
        return {
            ch for cond in r_conditions() if cond["file"] == filename
            for ch in cond["channels"]
        }

    @output
    @render.ui
    def manual_channels_ui():
        """Checkbox grid of the channels still unassigned for the selected
        file. Re-renders (and so resets the ticks) when the file changes or a
        condition is added/removed."""
        filename = input.manual_file()
        if not filename:
            return ui.p("Upload files and pick one above.", class_="text-muted small")

        available = [c for c in range(1, 33) if c not in _assigned_channels(filename)]
        if not available:
            return ui.p("All 32 channels are assigned for this file.",
                        class_="text-success small")
        return ui.input_checkbox_group(
            "manual_channels", "Channels (tick to include)",
            choices={str(c): str(c) for c in available},
            inline=True,
        )

    @output
    @render.data_frame
    def manual_conditions_table():
        conds = r_conditions()
        if not conds:
            return render.DataGrid(pd.DataFrame(
                {"info": ["No conditions yet — add one above."]}))
        table = pd.DataFrame([
            {"file": c["file"], "condition": c["label"],
             "channels": _compress_channels(list(c["channels"]))}
            for c in conds
        ])
        return render.DataGrid(table, width="100%", selection_mode="rows")

    @reactive.effect
    @reactive.event(input.dam_files)
    def _reset_conditions_on_upload():
        files = input.dam_files() or []
        names = [f["name"] for f in files]
        ui.update_select("manual_file", choices=names,
                         selected=names[0] if names else None)
        r_conditions.set([])

    @reactive.effect
    @reactive.event(input.manual_selectall_btn)
    def _select_all_channels():
        available = [c for c in range(1, 33)
                     if c not in _assigned_channels(input.manual_file())]
        ui.update_checkbox_group("manual_channels",
                                 selected=[str(c) for c in available])

    @reactive.effect
    @reactive.event(input.manual_clearsel_btn)
    def _clear_channel_ticks():
        ui.update_checkbox_group("manual_channels", selected=[])

    @reactive.effect
    @reactive.event(input.manual_add_btn)
    def _add_condition():
        filename = input.manual_file()
        name = (input.manual_cond_name() or "").strip()
        picked = input.manual_channels() or []

        if not filename:
            ui.notification_show("Upload files first.", type="warning")
            return
        if not name:
            ui.notification_show("Enter a condition name.", type="warning")
            return
        if not picked:
            ui.notification_show("Tick at least one channel.", type="warning")
            return

        channels = tuple(sorted(int(c) for c in picked))
        # Belt-and-braces: the grid only offers unassigned channels, but guard
        # against a stale selection racing a re-render.
        overlap = set(channels) & _assigned_channels(filename)
        if overlap:
            ui.notification_show(
                f"Channel(s) {sorted(overlap)} already assigned on this file.",
                type="error")
            return

        r_conditions.set(r_conditions() + [
            {"file": filename, "label": name, "channels": channels}
        ])
        ui.update_text("manual_cond_name", value="")

    @reactive.effect
    @reactive.event(input.manual_remove_btn)
    def _remove_selected_conditions():
        selection = manual_conditions_table.cell_selection()
        rows = set(selection.get("rows", ())) if selection else set()
        if not rows:
            ui.notification_show("Select row(s) in the table to remove.",
                                 type="warning")
            return
        r_conditions.set([c for i, c in enumerate(r_conditions()) if i not in rows])

    @reactive.effect
    @reactive.event(input.manual_clear_btn)
    def _clear_all_conditions():
        r_conditions.set([])

    @output
    @render.ui
    def load_status():
        store = r_store()
        if store is None:
            err = r_load_error()
            if err:
                return ui.p(f"Load failed: {err}", class_="text-danger fw-bold")
            return ui.p("No data loaded. Upload files and click Load data.",
                        class_="text-muted")
        n = len(store.animal_ids)
        labels = ", ".join(store.labels)
        return ui.div(
            ui.p(f"Loaded {n} animals across groups: {labels}",
                 class_="text-success fw-bold"),
        )

    @output
    @render.download(filename="metadata.xlsx")
    def save_metadata_btn():
        if not r_conditions():
            ui.notification_show("Add at least one condition first.", type="warning")
            return
        conditions = build_manual_conditions(_collect_manual_entries())
        tmp_path = Path(tempfile.mkstemp(suffix=".xlsx")[1])
        try:
            conditions.to_excel(tmp_path, index=False)
            yield tmp_path.read_bytes()
        finally:
            tmp_path.unlink(missing_ok=True)

    @output
    @render.data_frame
    def animals_table():
        store = r_store()
        if store is None:
            return render.DataGrid(pd.DataFrame())
        return render.DataGrid(
            store.metadata[["id", "monitor", "channel", "labels", "order"]]
            .sort_values(["order", "channel"]),
            width="100%",
        )

    # ── visual inspection ─────────────────────────────────────────────────────

    @output
    @render.ui
    def actogram_plot():
        df  = _data()
        aid = input.vis_animal()
        if not aid:
            return ui.p("Select an animal.", class_="text-muted")
        fig = plot_actogram(df, aid, bin_min=int(input.vis_bin_min()))
        return _fig_html(fig)

    @output
    @render.ui
    def daily_profile_plot():
        df = _data()
        fig = plot_daily_profile(df, bin_min=int(input.vis_profile_bin()))
        return _fig_html(fig)

    # ── activity analysis ─────────────────────────────────────────────────────

    @reactive.effect
    @reactive.event(input.act_analyse_btn)
    def _run_activity_bouts():
        df = _data()
        r_ab_df.set(detect_activity_bouts(df, threshold_min=input.act_bout_window()))

    @reactive.calc
    @reactive.event(input.act_plot_btn, input.act_analyse_btn)
    def _activity_summary():
        req(r_store() is not None)
        metric  = input.act_metric()
        groupby = input.act_groupby()

        if metric == "total":
            df = _data()
            fns = {"phase": activity_by_phase,
                   "day":   activity_by_day,
                   "day_phase": activity_by_day_phase}
        else:
            req(r_ab_df() is not None,
                cancel_output="Run bout detection first (click the button).")
            df = r_ab_df()
            if metric == "bout_activity":
                fns = {"phase": bout_activity_by_phase,
                       "day":   bout_activity_by_day,
                       "day_phase": bout_activity_by_day_phase}
            else:
                fns = {"phase": bout_duration_by_phase,
                       "day":   bout_duration_by_day,
                       "day_phase": bout_duration_by_day_phase}

        return fns[groupby](df)

    @output
    @render.ui
    def activity_plot():
        summary = _activity_summary()
        ylabs = {"total": "Total activity (counts)",
                 "bout_activity": "Mean activity per bout",
                 "bout_duration": "Mean bout duration (s)"}
        fig = box_plot(summary, ylab=ylabs.get(input.act_metric(), "y"))
        return _fig_html(fig)

    @output
    @render.data_frame
    def activity_stats():
        summary = _activity_summary()
        stats = summary_stats(summary)
        return render.DataGrid(stats.round(3), width="100%")

    # ── sleep analysis ────────────────────────────────────────────────────────

    @reactive.effect
    @reactive.event(input.sleep_analyse_btn)
    def _run_sleep():
        df = _data()
        annotated = annotate_sleep(df, threshold_min=input.sleep_threshold())
        with_bouts = detect_sleep_bouts(annotated)
        r_sleep.set(with_bouts)

    @reactive.calc
    @reactive.event(input.sleep_plot_btn, input.sleep_analyse_btn)
    def _sleep_summary():
        req(r_sleep() is not None,
            cancel_output="Run sleep annotation first (click the button).")
        df      = r_sleep()
        metric  = input.sleep_metric()
        groupby = input.sleep_groupby()

        dispatch = {
            ("ratio",         "phase"):     sleep_ratio_by_phase,
            ("ratio",         "day"):       sleep_ratio_by_day,
            ("ratio",         "day_phase"): sleep_ratio_by_day_phase,
            ("tst",           "phase"):     lambda d: tst_mean(d),
            ("tst",           "day"):       tst_by_day,
            ("tst",           "day_phase"): tst_by_day,
            ("waso",          "phase"):     lambda d: waso_mean(d),
            ("waso",          "day"):       waso_by_day,
            ("waso",          "day_phase"): waso_by_day,
            ("latency",       "phase"):     lambda d: sleep_latency_mean(d),
            ("latency",       "day"):       sleep_latency_by_day,
            ("latency",       "day_phase"): sleep_latency_by_day,
            ("bout_duration", "phase"):     sleep_bout_duration_by_phase,
            ("bout_duration", "day"):       sleep_bout_duration_by_day,
            ("bout_duration", "day_phase"): sleep_bout_duration_by_day,
            ("bout_count",    "phase"):     sleep_bout_count_by_phase,
            ("bout_count",    "day"):       sleep_bout_count_by_day,
            ("bout_count",    "day_phase"): sleep_bout_count_by_day,
        }
        fn = dispatch.get((metric, groupby))
        return fn(df) if fn else pd.DataFrame()

    @output
    @render.ui
    def sleep_plot():
        summary = _sleep_summary()
        if summary.empty:
            return ui.p("No data — run sleep annotation first.",
                        class_="text-muted")
        ylabs = {
            "ratio":         "Sleep ratio (fraction)",
            "tst":           "TST (h)",
            "waso":          "WASO (h)",
            "latency":       "Sleep latency (min)",
            "bout_duration": "Bout duration (min)",
            "bout_count":    "Bout count",
        }
        fig = box_plot(summary, ylab=ylabs.get(input.sleep_metric(), "y"))
        return _fig_html(fig)

    @output
    @render.data_frame
    def sleep_stats():
        summary = _sleep_summary()
        if summary.empty:
            return render.DataGrid(pd.DataFrame())
        stats = summary_stats(summary)
        return render.DataGrid(stats.round(3), width="100%")

    # ── rhythm analysis ───────────────────────────────────────────────────────

    @reactive.effect
    @reactive.event(input.rhythm_btn)
    def _run_rhythm():
        df      = _data()
        method  = input.rhythm_method()
        periods = np.arange(
            float(input.rhythm_period_from()),
            float(input.rhythm_period_to()) + 0.001,
            float(input.rhythm_period_step()),
        )
        pg = periodogram_by_animal(df, method=method, periods_h=periods)
        r_pgram.set(pg)
        r_cosinor.set(cosinor_summary(df))
        r_isivra.set(is_iv_ra(df))

    @output
    @render.ui
    def rhythm_plot():
        view = input.rhythm_view()

        if view == "periodogram":
            pg = r_pgram()
            if pg is None:
                return ui.p("Click Run analysis.", class_="text-muted")
            value_col = "power" if input.rhythm_method() == "lomb_scargle" else "Q"
            fig = periodogram_plot(pg, value_col=value_col)
            return _fig_html(fig)

        elif view == "dominant_period":
            pg = r_pgram()
            if pg is None:
                return ui.p("Click Run analysis.", class_="text-muted")
            value_col = "power" if input.rhythm_method() == "lomb_scargle" else "Q"
            dp = dominant_period_by_animal(_data())
            fig = box_plot(dp, ylab="Dominant period (h)",
                           title="Dominant Circadian Period")
            return _fig_html(fig)

        elif view == "cosinor":
            cos = r_cosinor()
            if cos is None:
                return ui.p("Click Run analysis.", class_="text-muted")
            fig = box_plot(cos, ylab="Value", title="Cosinor Parameters")
            return _fig_html(fig)

        elif view == "is_iv_ra":
            isivra = r_isivra()
            if isivra is None:
                return ui.p("Click Run analysis.", class_="text-muted")
            fig = box_plot(isivra, ylab="Value",
                           title="Non-parametric Circadian Metrics")
            return _fig_html(fig)

        return ui.p("Select a view.", class_="text-muted")

    @output
    @render.data_frame
    def rhythm_table():
        view = input.rhythm_view()

        if view == "dominant_period":
            pg = r_pgram()
            if pg is None:
                return render.DataGrid(pd.DataFrame())
            dp = dominant_period_by_animal(_data())
            return render.DataGrid(
                dp[["id", "labels", "y"]].rename(columns={"y": "dominant_period_h"})
                .round(2), width="100%")

        elif view == "cosinor":
            cos = r_cosinor()
            if cos is None:
                return render.DataGrid(pd.DataFrame())
            wide = cos.pivot_table(
                index=["id", "labels"], columns="x", values="y"
            ).reset_index()
            return render.DataGrid(wide.round(3), width="100%")

        elif view == "is_iv_ra":
            isivra = r_isivra()
            if isivra is None:
                return render.DataGrid(pd.DataFrame())
            wide = isivra.pivot_table(
                index=["id", "labels"], columns="x", values="y"
            ).reset_index()
            return render.DataGrid(wide.round(3), width="100%")

        return render.DataGrid(pd.DataFrame())

    # ── survival analysis ─────────────────────────────────────────────────────

    @reactive.effect
    @reactive.event(input.surv_btn)
    def _run_survival():
        df     = _data()
        window = float(input.dead_window_h())
        surv   = build_survival_data(df, dead_window_h=window)
        r_surv_df.set(surv)
        r_km_df.set(kaplan_meier_curves(surv))
        r_flagged.set(flag_ambiguous_deaths(df, dead_window_h=window))

    def _no_deaths() -> bool:
        surv = r_surv_df()
        return surv is not None and int(surv["event"].sum()) == 0

    @output
    @render.ui
    def km_plot():
        surv = r_surv_df()
        km = r_km_df()
        if surv is None or km is None:
            return ui.p("Click Run survival analysis.", class_="text-muted")
        if _no_deaths():
            return ui.div(
                ui.p("There is no death animal detected at this silence window.",
                     class_="text-warning fw-bold"),
                ui.p("All animals stayed active through the recording (they are "
                     "right-censored). If you expect deaths, try a shorter "
                     "silence window or a longer recording.",
                     class_="text-muted small"),
            )
        return _fig_html(plot_km(km))

    @output
    @render.data_frame
    def median_surv_table():
        surv = r_surv_df()
        if surv is None:
            return render.DataGrid(pd.DataFrame())
        if _no_deaths():
            return render.DataGrid(pd.DataFrame(
                {"status": ["No death animal detected — median survival is undefined."]}))
        df = median_survival(surv)[["labels", "x", "y", "ci_lower", "ci_upper"]]
        df = df.rename(columns={"x": "group", "y": "median_days"}).round(2)
        return render.DataGrid(df, width="100%")

    @output
    @render.data_frame
    def logrank_table():
        surv = r_surv_df()
        if surv is None:
            return render.DataGrid(pd.DataFrame())
        if _no_deaths():
            return render.DataGrid(pd.DataFrame(
                {"status": ["No death animal detected — log-rank test is not applicable."]}))
        return render.DataGrid(
            logrank_test(surv).round({"test_statistic": 3, "p_value": 6}),
            width="100%",
        )

    @output
    @render.data_frame
    def flagged_deaths_table():
        flagged = r_flagged()
        if flagged is None:
            return render.DataGrid(pd.DataFrame())
        if flagged.empty:
            return render.DataGrid(pd.DataFrame(
                {"status": ["No animals moved after their inferred death."]}
            ))
        return render.DataGrid(
            flagged.rename(columns={
                "labels": "condition",
                "death_day": "inferred_death_day",
                "post_death_counts": "activity_after_death",
                "post_death_active_bins": "active_bins_after_death",
                "last_active_day": "last_movement_day",
            }),
            width="100%",
        )

    # ── metric downloads (R-style multi-sheet .xlsx) ──────────────────────────

    def _settings_rows() -> list[tuple[str, object]]:
        return [
            ("LD cycle (h)", float(input.ld_period())),
            ("Light hours per cycle (h)", float(input.light_hours())),
            ("Sleep threshold (min)", int(input.sleep_threshold_setting())),
            ("Activity bout gap threshold (min)", int(input.bout_threshold_setting())),
            ("Light onset (ZT0)", input.zt0()),
        ]

    def _dates_by_id() -> dict[str, tuple[str, str]]:
        df = _data()
        return {
            aid: (str(s.min()), str(s.max()))
            for aid, s in df.groupby("id", sort=False)["datetime"]
        }

    def _yield_xlsx(write_fn):
        """Run write_fn(tmp_path) and stream the resulting file, then clean up."""
        tmp = Path(tempfile.mkstemp(suffix=".xlsx")[1])
        try:
            write_fn(tmp)
            yield tmp.read_bytes()
        finally:
            tmp.unlink(missing_ok=True)

    def _safe(fetch):
        """Fetch a summary, swallowing the reactive 'not computed yet' signal."""
        try:
            return fetch()
        except Exception:
            return None

    @output
    @render.download(filename=lambda: "activity_metric.xlsx")
    def dl_activity():
        summary = _safe(_activity_summary)
        if summary is None or summary.empty:
            ui.notification_show("Update the plot first, then download.",
                                 type="warning")
            return
        yield from _yield_xlsx(lambda p: write_metric_workbook(
            p, summary, _settings_rows(), _dates_by_id()))

    @output
    @render.download(filename=lambda: "sleep_metric.xlsx")
    def dl_sleep():
        summary = _safe(_sleep_summary)
        if summary is None or summary.empty:
            ui.notification_show("Run sleep annotation and update the plot first.",
                                 type="warning")
            return
        yield from _yield_xlsx(lambda p: write_metric_workbook(
            p, summary, _settings_rows(), _dates_by_id()))

    @output
    @render.download(filename=lambda: "rhythm_metric.xlsx")
    def dl_rhythm():
        view = input.rhythm_view()
        if view == "cosinor":
            summary = r_cosinor()
        elif view == "is_iv_ra":
            summary = r_isivra()
        elif view == "dominant_period":
            summary = _safe(lambda: dominant_period_by_animal(_data())
                            .assign(x="Dominant period (h)"))
        else:  # periodogram — a curve per animal, not the standard summary schema
            ui.notification_show(
                "Switch the View to Cosinor, IS/IV/RA, or Dominant period to "
                "download a metric table.", type="warning")
            return
        if summary is None or summary.empty:
            ui.notification_show("Click Run analysis first.", type="warning")
            return
        yield from _yield_xlsx(lambda p: write_metric_workbook(
            p, summary, _settings_rows(), _dates_by_id()))

    @output
    @render.download(filename=lambda: "survival.xlsx")
    def dl_survival():
        surv = r_surv_df()
        if surv is None:
            ui.notification_show("Run survival analysis first.", type="warning")
            return

        def _write(path):
            settings_df = pd.DataFrame(
                _settings_rows() + [("Silence window to call death (h)",
                                     float(input.dead_window_h()))],
                columns=["Settings", "Values"])
            per_animal = surv[["id", "labels", "channel", "monitor",
                               "duration", "event"]].rename(columns={
                "labels": "condition", "duration": "duration_days",
                "event": "died"})
            with pd.ExcelWriter(path, engine="openpyxl") as xw:
                settings_df.to_excel(xw, sheet_name="Settings", index=False)
                per_animal.to_excel(xw, sheet_name="Per animal", index=False)
                if not _no_deaths():
                    median_survival(surv).rename(
                        columns={"x": "group", "y": "median_days"}
                    ).to_excel(xw, sheet_name="Median survival", index=False)
                    logrank_test(surv).to_excel(
                        xw, sheet_name="Log-rank", index=False)
                flagged = r_flagged()
                if flagged is not None and not flagged.empty:
                    flagged.to_excel(xw, sheet_name="Flagged animals", index=False)

        yield from _yield_xlsx(_write)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

app = App(app_ui, server)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
