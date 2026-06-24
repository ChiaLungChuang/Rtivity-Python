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

from modules.data.data_store import DataStore, Settings
from modules.analysis.activity import (
    activity_by_day,
    activity_by_day_phase,
    activity_by_phase,
    bout_activity_by_day,
    bout_activity_by_phase,
    bout_duration_by_day,
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
    kaplan_meier_curves,
    logrank_test,
    median_survival,
)
from modules.ui.plots import (
    actogram_plot,
    box_plot,
    daily_profile_plot,
    km_plot,
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
                ui.input_file("metadata_file", "Metadata file (.xlsx)",
                              accept=[".xlsx"]),
                ui.hr(),
                ui.h6("Settings", class_="fw-bold"),
                ui.input_text("zt0", "Lights-on time (ZT0)", value="08:00:00"),
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

    # ── Tab 2 · Visual inspection ─────────────────────────────────────────────
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

    # ── Tab 3 · Activity analysis ─────────────────────────────────────────────
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
            ),
        ),
    ),

    # ── Tab 4 · Sleep analysis ────────────────────────────────────────────────
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
            ),
        ),
    ),

    # ── Tab 5 · Rhythm analysis ───────────────────────────────────────────────
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
            ),
        ),
    ),

    # ── Tab 6 · Survival analysis ─────────────────────────────────────────────
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

    r_store   = reactive.value(None)   # DataStore
    r_ab_df   = reactive.value(None)   # data with activity bouts
    r_sleep   = reactive.value(None)   # data with sleep annotation + bouts
    r_pgram   = reactive.value(None)   # periodogram DataFrame
    r_cosinor = reactive.value(None)   # cosinor DataFrame
    r_isivra  = reactive.value(None)   # IS/IV/RA DataFrame
    r_surv_df = reactive.value(None)   # survival data
    r_km_df   = reactive.value(None)   # KM curves

    # ── data loading ──────────────────────────────────────────────────────────

    @reactive.effect
    @reactive.event(input.load_btn)
    def _load():
        dam_files  = input.dam_files()
        meta_files = input.metadata_file()
        if not dam_files or not meta_files:
            r_store.set(None)
            return

        try:
            tmpdir = Path(tempfile.mkdtemp(prefix="rtivity_"))
            for f in dam_files:
                shutil.copy(f["datapath"], tmpdir / f["name"])

            meta_path = tmpdir / "metadata.xlsx"
            shutil.copy(meta_files[0]["datapath"], meta_path)

            settings = Settings(
                zt0=input.zt0(),
                ld_period=float(input.ld_period()),
                light_hours=float(input.light_hours()),
                sleep_threshold_min=int(input.sleep_threshold_setting()),
                bout_threshold_min=int(input.bout_threshold_setting()),
            )
            store = DataStore(dam_dir=tmpdir, settings=settings)
            store.load_from_metadata(meta_path)

            # Invalidate all derived caches
            r_ab_df.set(None)
            r_sleep.set(None)
            r_pgram.set(None)
            r_cosinor.set(None)
            r_isivra.set(None)
            r_surv_df.set(None)
            r_km_df.set(None)

            r_store.set(store)

            # Populate animal / group selectors
            animals = store.animal_ids
            labels  = store.labels
            ui.update_select("vis_animal",  choices=animals,
                             selected=animals[0] if animals else None)
            ui.update_select("vis_group",   choices=labels,
                             selected=labels[0] if labels else None)

        except Exception:
            r_store.set(None)
            print(traceback.format_exc())

    # ── derived data accessors ────────────────────────────────────────────────

    @reactive.calc
    def _data():
        req(r_store() is not None)
        return r_store().data

    # ── load status + animals table ───────────────────────────────────────────

    @output
    @render.ui
    def load_status():
        store = r_store()
        if store is None:
            return ui.p("No data loaded. Upload files and click Load data.",
                        class_="text-muted")
        n = len(store.animal_ids)
        labels = ", ".join(store.labels)
        return ui.div(
            ui.p(f"Loaded {n} animals across groups: {labels}",
                 class_="text-success fw-bold"),
        )

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
        fig = actogram_plot(df, aid, bin_min=int(input.vis_bin_min()))
        return _fig_html(fig)

    @output
    @render.ui
    def daily_profile_plot():
        df = _data()
        fig = daily_profile_plot(df, bin_min=int(input.vis_profile_bin()))
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
                       "day_phase": lambda d: bout_activity_by_day(d)}
            else:
                fns = {"phase": bout_duration_by_phase,
                       "day":   bout_duration_by_day,
                       "day_phase": lambda d: bout_duration_by_day(d)}

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
        df   = _data()
        surv = build_survival_data(df, dead_window_h=float(input.dead_window_h()))
        r_surv_df.set(surv)
        r_km_df.set(kaplan_meier_curves(surv))

    @output
    @render.ui
    def km_plot():
        km = r_km_df()
        if km is None:
            return ui.p("Click Run survival analysis.", class_="text-muted")
        return _fig_html(km_plot(km))

    @output
    @render.data_frame
    def median_surv_table():
        surv = r_surv_df()
        if surv is None:
            return render.DataGrid(pd.DataFrame())
        df = median_survival(surv)[["labels", "x", "y", "ci_lower", "ci_upper"]]
        df = df.rename(columns={"x": "group", "y": "median_days"}).round(2)
        return render.DataGrid(df, width="100%")

    @output
    @render.data_frame
    def logrank_table():
        surv = r_surv_df()
        if surv is None:
            return render.DataGrid(pd.DataFrame())
        return render.DataGrid(
            logrank_test(surv).round({"test_statistic": 3, "p_value": 6}),
            width="100%",
        )


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

app = App(app_ui, server)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
