from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PaperConfig:
    """Configuration for high-resolution paper output"""

    input_dir: Path = Path("pivot_tables_csv")
    output_dir: Path = Path("radar_plots_paper")
    paper_width_px: int = 700
    paper_height_px: int = 400

    # Fonts
    font_family: str = "Arial, sans-serif"
    base_font_size: int = 10
    category_font_size: int = 11
    legend_font_size: int = 13
    title_font_size: int = 13
    radial_tick_font_size: int = 11

    # Trace appearance
    line_width: float = 2.5
    fill_opacity: float = 0.18
    line_color_darkness: float = 0.75

    # Legend
    legend_border_width: int = 1
    legend_border_color: str = "rgba(0,0,0,0.1)"

    # Output
    dpi: int = 3
    export_formats: tuple = ("png",)


@dataclass(frozen=True)
class WebConfig:
    """Configuration for web-responsive output"""

    input_dir: Path = Path("pivot_tables_csv")
    output_dir: Path = Path("radar_plots_web")
    paper_width_px: int = 700
    paper_height_px: int = 400

    # Fonts
    font_family: str = "Arial, sans-serif"
    base_font_size: int = 10
    category_font_size: int = 11
    legend_font_size: int = 9
    title_font_size: int = 13
    radial_tick_font_size: int = 8

    # Trace appearance
    line_width: float = 2.5
    fill_opacity: float = 0.18
    line_color_darkness: float = 0.75

    # Legend
    legend_border_width: int = 1
    legend_border_color: str = "rgba(0,0,0,0.1)"

    # Output
    dpi: int = 1
    export_formats: tuple = ("html",)


CATEGORY_LABELS: Dict[str, str] = {
    "commonsense": "Common<br>Sense",
    "culture": "Culture",
    "deduction": "Logic",
    "knowledge_and_qa": "Knowledge",
    "language_and_communication": "Language",
    "liberal_arts": "Liberal<br>Arts",
    "math": "Math",
    "safety_and_truthfulness": "Safety",
    "science,technology,engineering": "Science/<br>Tech",
    "instruction_following": "Instruction",
}

MODEL_NAMES: Dict[str, str] = {
    "GAIR/LIMO": "LIMO",
    "GAIR/LIMO-v2": "LIMO v2",
    "PrimeIntellect/INTELLECT-2": "INTELLECT-2",
    "Qwen/QwQ-32B": "QwQ (32B)",
    "Qwen/Qwen2.5-3B-Instruct": "Qwen 2.5 Instruct (3B)",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen 2.5 Instruct (7B)",
    "Qwen/Qwen2.5-14B-Instruct": "Qwen 2.5 Instruct (14B)",
    "Qwen/Qwen2.5-32B-Instruct": "Qwen 2.5 Instruct (32B)",
    "Qwen/Qwen2.5-72B-Instruct": "Qwen 2.5 Instruct (72B)",
    "Qwen/Qwen2.5-Coder-3B": "Qwen 2.5 Coder (3B)",
    "Qwen/Qwen2.5-Coder-7B": "Qwen 2.5 Coder (7B)",
    "Qwen/Qwen2.5-Coder-14B": "Qwen 2.5 Coder (14B)",
    "Qwen/Qwen2.5-Coder-32B": "Qwen 2.5 Coder (32B)",
    "Qwen/Qwen2.5-Coder-3B-Instruct": "Qwen 2.5 Coder Instruct (3B)",
    "Qwen/Qwen2.5-Coder-7B-Instruct": "Qwen 2.5 Coder Instruct (7B)",
    "Qwen/Qwen2.5-Coder-14B-Instruct": "Qwen 2.5 Coder Instruct (14B)",
    "Qwen/Qwen2.5-Coder-32B-Instruct": "Qwen 2.5 Coder Instruct (32B)",
    "Qwen/Qwen2.5-Math-7B": "Qwen 2.5 Math (7B)",
    "Qwen/Qwen2.5-Math-7B-Instruct": "Qwen 2.5 Math Instruct (7B)",
    "Qwen/Qwen3-8B": "Qwen3 (8B)",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": "R1 Distill Llama (8B)",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": "R1 Distill Qwen (1.5B)",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": "R1 Distill Qwen (7B)",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": "R1 Distill Qwen (32B)",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama 3.1 Instruct (8B)",
    "meta-llama/Llama-3.2-3B-Instruct": "Llama 3.2 Instruct (3B)",
    "NousResearch/Hermes-3-Llama-3.1-8B": "Hermes3 (8B)",
    "nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1": "Nemotron Nano (4B)",
    "nvidia/Llama-3.1-Nemotron-Nano-8B-v1.1": "Nemotron Nano (8B)",
    "nvidia/Llama-3.1-Nemotron-Nano-32B-v1.1": "Nemotron Nano (32B)",
    "nvidia/OpenCodeReasoning-Nemotron-1.1-7B": "Nemotron Code Reasoning (7B)",
    "nvidia/OpenCodeReasoning-Nemotron-1.1-14B": "Nemotron Code Reasoning (14B)",
    "nvidia/OpenMath2-Llama3.1-8B": "Open Math 2 (8B)",
    "nvidia/OpenReasoning-Nemotron-7B": "OpenReasoning (7B)",
    "nvidia/OpenReasoning-Nemotron-14B": "Nemotron (14B)",
    "nvidia/OpenReasoning-Nemotron-32B": "OpenReasoning (32B)",
    "open-thoughts/OpenThinker-7B": "OpenThinker (7B)",
    "open-thoughts/OpenThinker-8B-v1": "OpenThinker (8B)",
    "open-thoughts/OpenThinker2-32B": "OpenThinker2 (32B)",
    "open-thoughts/OpenThinker3-7B": "OpenThinker3 (7B)",
    "simplescaling/s1.1-7B": "s1.1 (7B)",
    "simplescaling/s1.1-14B": "s1.1 (14B)",
    "simplescaling/s1.1-32B": "s1.1 (32B)",
}

# --------------------------
# Color palette
# --------------------------
COLOR_PALETTE: List[str] = [
    "#E60049", "#0BB4FF", "#50E991", "#F1C40F", "#9B19F5",
    "#FFA300", "#DC0AB4", "#B3D4FF", "#00BFA0", "#F15F5F",
]


# --------------------------
# Helpers
# --------------------------

def clean_name(name: str) -> str:
    """Normalize column names using MODEL_NAMES and small text fixes."""
    simplified = name.replace(" → ", " -> ").replace("_", "/").split(" -> ")[-1].strip()
    return MODEL_NAMES.get(simplified, simplified)


def load_data(path: Path) -> Optional[pd.DataFrame]:
    """Load a CSV pivot table and prepare it for plotting."""
    if not path.exists():
        log.debug("Missing file: %s", path)
        return None

    try:
        df = pd.read_csv(path, index_col=0)
        df = df[~df.index.isin(["OVERALL", "Missing"])].copy()
        df.columns = [clean_name(c) for c in df.columns]
        return df if not df.empty else None
    except Exception as exc:
        log.warning("Error loading %s: %s", path, exc)
        return None


def get_colors(n: int) -> List[str]:
    """Return n distinct colors cycling through the palette."""
    return [c for c, _ in zip(cycle(COLOR_PALETTE), range(n))]


def darken_hex(hex_color: str, factor: float) -> str:
    """Return an rgb(...) string with the hex color darkened by factor (0..1)."""
    hex_color = hex_color.lstrip("#")
    r = int(int(hex_color[0:2], 16) * factor)
    g = int(int(hex_color[2:4], 16) * factor)
    b = int(int(hex_color[4:6], 16) * factor)
    return f"rgb({r},{g},{b})"


def hex_to_rgba(hex_color: str, opacity: float) -> str:
    """Convert a hex color string to an rgba string."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(int(hex_color[4:6], 16))
    return f"rgba({r},{g},{b},{opacity})"


# --------------------------
# Plotting
# --------------------------

def _add_trace(
    fig: go.Figure,
    model: str,
    values: List[float],
    categories: List[str],
    color: str,
    config: PaperConfig | WebConfig,
    show_legend: bool = True,
    row: int = None,
    col: int = None,
) -> None:
    """Add a single radar trace to an existing figure."""
    r = values + [values[0]]
    theta = categories + [categories[0]]
    hovertemplate = (
        "<b>%{text}</b><br>Score: %{r:.1%}<br>Category: %{theta}<extra></extra>"
    )

    fig.add_trace(
        go.Scatterpolar(
            r=r,
            theta=theta,
            name=model,
            line=dict(
                color=darken_hex(color, config.line_color_darkness),
                width=config.line_width,
                shape='linear'
            ),
            marker=dict(symbol='circle', size=6),
            fill="toself",
            fillcolor=hex_to_rgba(color, config.fill_opacity),
            showlegend=show_legend,
            hovertemplate=hovertemplate,
            text=[model] * len(r),
        ),
        row=row,
        col=col,
    )


def _compute_ticks() -> Tuple[List[float], List[str], float]:
    """
    Compute fixed tick values and labels for the radial axis.
    The axis maximum is fixed at 50% for consistent comparisons.
    """
    tick_values = [0.15, 0.30, 0.45]
    tick_text = [f"{int(v * 100)}%" for v in tick_values]
    y_max = 0.50
    return tick_values, tick_text, y_max


def create_dual_plot(
    df1: Optional[pd.DataFrame],
    df2: Optional[pd.DataFrame],
    categories: List[str],
    config: PaperConfig | WebConfig,
) -> go.Figure:
    """Create a horizontally aligned pair of radar charts with standard axes."""
    tick_values, tick_text, y_max = _compute_ticks()

    # **MODIFIED: Removed subplot_titles argument**
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "polar"}, {"type": "polar"}]],
        horizontal_spacing=0.18,
    )

    all_models = sorted(set(
        (df1.columns.tolist() if df1 is not None else []) +
        (df2.columns.tolist() if df2 is not None else [])
    ))
    color_map = {m: c for m, c in zip(all_models, get_colors(len(all_models)))}

    # Add traces for left and right plots
    if df1 is not None:
        for m in df1.columns:
            _add_trace(fig, m, df1[m].fillna(0).tolist(), categories,
                       color_map.get(m, "#808080"), config, True, 1, 1)
    if df2 is not None:
        for m in df2.columns:
            _add_trace(fig, m, df2[m].fillna(0).tolist(), categories,
                       color_map.get(m, "#808080"), config, False, 1, 2)

    for trace in fig.data:
        trace.legendgroup = trace.name

    # Unified layout settings
    fig.update_layout(
        width=config.paper_width_px,
        height=config.paper_height_px,
        font=dict(family=config.font_family, size=config.base_font_size, color="black"),
        # **MODIFIED: Changed top margin `t` from 55 back to 35**
        margin=dict(l=60, r=60, t=35, b=85),
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5,
            font=dict(size=config.legend_font_size),
            bordercolor=config.legend_border_color,
            borderwidth=config.legend_border_width,
            itemwidth=35,
        ),
        template="plotly_white",
    )

    # Shared polar settings for both plots
    polar_settings = dict(
        angularaxis=dict(
            tickfont=dict(size=config.category_font_size),
            direction="clockwise", rotation=90,
            tickprefix="<b>", ticksuffix="</b>",
            gridcolor="darkgray",
        ),
        radialaxis=dict(
            range=[0, y_max],
            tickvals=tick_values,
            ticktext=tick_text,
            tickfont=dict(size=config.radial_tick_font_size),
            showline=False,
            gridcolor="darkgray"
        ),
    )

    fig.update_layout(polar=polar_settings, polar2=polar_settings)
    return fig


# --------------------------
# IO / Save helpers
# --------------------------

def save_figure(
    fig: go.Figure, path: Path, config: PaperConfig | WebConfig
) -> None:
    """Save a figure based on the configuration type."""
    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("  -> Saving %s", path.parent / f"{path.name}.*")

    for ext in config.export_formats:
        out = path.with_suffix("." + ext)
        try:
            if ext == "html":
                fig.write_html(
                    str(out),
                    config={
                        'responsive': True, 'displayModeBar': True, 'displaylogo': False,
                        'modeBarButtonsToRemove': ['lasso2d', 'select2d', 'toImage']
                    },
                    include_plotlyjs='cdn',
                )
            else:
                fig.write_image(str(out), scale=config.dpi)
        except Exception as exc:
            log.warning("Could not save %s: %s", out, exc)


# --------------------------
# Main processing logic
# --------------------------

def process_group(
    group: str, config: PaperConfig | WebConfig
) -> None:
    """Load all data for a group and generate all applicable plot types."""
    log.info("\nProcessing group: %s", group)

    input_dir = config.input_dir
    df_forget_orig = load_data(input_dir / f"{group}_Adj_Abs_Forgetting.csv")

    improve_path = input_dir / f"{group}_Adj_Abs_Backward-Transfer.csv"
    if not improve_path.exists():
        improve_path = input_dir / f"{group}_Adj_Abs_Improvement.csv"
    df_improve_orig = load_data(improve_path)

    df_acc_after_orig = load_data(input_dir / f"{group}_acc_after.csv")
    df_acc_before_orig = load_data(input_dir / f"{group}_acc_before.csv")

    # Loop to create full plots and "_base" plots (with some categories removed)
    for suffix, drop_list in [
        ("", []),
        ("_base", ["safety_and_truthfulness", "instruction_following"])
    ]:
        df_forget = df_forget_orig.copy() if df_forget_orig is not None else None
        df_improve = df_improve_orig.copy() if df_improve_orig is not None else None
        df_acc_after = df_acc_after_orig.copy() if df_acc_after_orig is not None else None
        df_acc_before = df_acc_before_orig.copy() if df_acc_before_orig is not None else None

        if drop_list:
            for df in [df_forget, df_improve, df_acc_after, df_acc_before]:
                if df is not None:
                    df.drop(index=drop_list, errors="ignore", inplace=True)

        # Plot 1: Forgetting vs. Backward-Transfer
        if (df_forget is not None and not df_forget.empty) or \
           (df_improve is not None and not df_improve.empty):
            df_ref = df_forget if (df_forget is not None and not df_forget.empty) else df_improve
            categories = [CATEGORY_LABELS.get(c, c) for c in df_ref.index]
            fig = create_dual_plot(df_forget, df_improve, categories, config)
            out_path = config.output_dir / "forget_vs_transfer" / f"{group}_side_by_side{suffix}"
            save_figure(fig, out_path, config)
        else:
            log.debug("  Skipping 'Forget vs Transfer' plot for %s%s (missing data).", group, suffix)

        # Plot 2: Accuracy Delta vs. Forgetting
        if (df_acc_after is not None and not df_acc_after.empty) and \
           (df_acc_before is not None and not df_acc_before.empty) and \
           (df_forget is not None and not df_forget.empty):

            # Align columns and indices before subtraction
            common_cols = df_acc_after.columns.intersection(df_acc_before.columns)
            common_idx = df_acc_after.index.intersection(df_acc_before.index)

            df_delta = df_acc_before.loc[common_idx, common_cols] - df_acc_after.loc[common_idx, common_cols]

            # Clip negative values to zero
            df_delta = df_delta.clip(lower=0)

            # Align the forgetting dataframe to match the delta dataframe
            df_forget_aligned = df_forget.loc[common_idx, common_cols]

            categories = [CATEGORY_LABELS.get(c, c) for c in df_delta.index]

            fig = create_dual_plot(df_delta, df_forget_aligned, categories, config)
            out_path = config.output_dir / "delta_vs_forgetting" / f"{group}_delta_vs_forgetting{suffix}"
            save_figure(fig, out_path, config)
        else:
            log.debug("  Skipping 'Delta vs Forget' plot for %s%s (missing data).", group, suffix)


# --------------------------
# CLI and entrypoint
# --------------------------

def main(argv: List[str] | None = None) -> None:
    import argparse
    import re

    parser = argparse.ArgumentParser(description="Generate radar plots from pivot-table CSVs.")
    parser.add_argument("--input-dir", type=Path, default=Path("pivot_tables_csv"), help="Directory containing pivot CSVs")
    parser.add_argument("--output-dir", type=Path, help="Base directory to save plots (auto-generated if not specified)")
    parser.add_argument("--mode", choices=["paper", "web", "both"], default="both", help="Generate plots for paper, web, or both")
    parser.add_argument("--dpi", type=int, default=3, help="Scale factor (DPI) for paper exports")
    parser.add_argument("--formats", nargs="*", default=["png"], help="Output formats for paper (e.g., png svg pdf)")
    args = parser.parse_args(argv)

    # Find all unique group names from relevant CSV files
    csvs = list(args.input_dir.glob("*.csv"))
    groups = set()
    pattern = re.compile(r"(.+?)_((Adj_Abs|Max_Possible)_(Forgetting|Improvement|Backward-Transfer)|acc_before|acc_after)\.csv")
    for p in csvs:
        match = pattern.match(p.name)
        if match:
            groups.add(match.group(1))

    sorted_groups = sorted(list(groups))
    if not sorted_groups:
        log.warning("No matching CSVs found in %s", args.input_dir)
        return

    log.info("Found %d groups: %s", len(sorted_groups), ", ".join(sorted_groups))

    def run_for_config(base_cfg):
        base_cfg.output_dir.mkdir(parents=True, exist_ok=True)
        for g in sorted_groups:
            process_group(g, base_cfg)

    # Generate paper versions
    if args.mode in ["paper", "both"]:
        log.info("\n" + "=" * 60 + "\nGENERATING PAPER VERSIONS\n" + "=" * 60)
        paper_cfg = PaperConfig(
            input_dir=args.input_dir,
            output_dir=args.output_dir or Path("radar_plots_paper"),
            dpi=args.dpi,
            export_formats=tuple(args.formats)
        )
        run_for_config(paper_cfg)

    # Generate web versions
    if args.mode in ["web", "both"]:
        log.info("\n" + "=" * 60 + "\nGENERATING WEB VERSIONS\n" + "=" * 60)
        web_cfg = WebConfig(
            input_dir=args.input_dir,
            output_dir=args.output_dir or Path("radar_plots_web"),
        )
        run_for_config(web_cfg)

    log.info("\nAll plots generated successfully!")


if __name__ == "__main__":
    main()
