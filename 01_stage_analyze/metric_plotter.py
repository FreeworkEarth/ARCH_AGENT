#!/usr/bin/env python3
"""
Metric Plotter - Time-series visualization for DV8 temporal analysis.

Generates matplotlib plots showing architecture metric evolution over time with:
- Threshold zones (good/warning/bad ranges)
- Annotations for significant changes
- Combined overview of all metrics
- Individual detailed plots per metric
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

try:
    import matplotlib
    # Force non-interactive backend for headless environments
    try:
        matplotlib.use('Agg')
    except Exception:
        pass
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle
except ImportError:
    print("ERROR: matplotlib not installed. Install with:")
    print("  pip3 install matplotlib")
    sys.exit(1)


# --- Metric Configuration ---

METRIC_CONFIG = {
    "propagation-cost": {
        "title": "Propagation Cost",
        "ylabel": "Propagation Cost (%)",
        "description": "How changes cascade through the system",
        "thresholds": {
            "excellent": (0, 30),
            "good": (30, 50),
            "moderate": (50, 70),
            "poor": (70, 100)
        },
        "colors": {
            "excellent": "#d4edda",  # light green
            "good": "#fff3cd",       # light yellow
            "moderate": "#f8d7da",   # light red
            "poor": "#f5c6cb"        # darker red
        },
        "invert": False  # lower is better
    },
    "m-score": {
        "title": "M-Score (Modularity Quality)",
        "ylabel": "M-Score (%)",
        "description": "Overall modularity quality",
        "thresholds": {
            "poor": (0, 30),
            "moderate": (30, 50),
            "good": (50, 70),
            "excellent": (70, 100)
        },
        "colors": {
            "poor": "#f5c6cb",
            "moderate": "#f8d7da",
            "good": "#fff3cd",
            "excellent": "#d4edda"
        },
        "invert": True  # higher is better
    },
    "decoupling-level": {
        "title": "Decoupling Level",
        "ylabel": "Decoupling Level (%)",
        "description": "Independence of modules",
        "thresholds": {
            "poor": (0, 30),
            "moderate": (30, 50),
            "good": (50, 70),
            "excellent": (70, 100)
        },
        "colors": {
            "poor": "#f5c6cb",
            "moderate": "#f8d7da",
            "good": "#fff3cd",
            "excellent": "#d4edda"
        },
        "invert": True
    },
    "independence-level": {
        "title": "Independence Level",
        "ylabel": "Independence Level (%)",
        "description": "Files depending only on design rules",
        "thresholds": {
            "poor": (0, 20),
            "moderate": (20, 40),
            "good": (40, 60),
            "excellent": (60, 100)
        },
        "colors": {
            "poor": "#f5c6cb",
            "moderate": "#f8d7da",
            "good": "#fff3cd",
            "excellent": "#d4edda"
        },
        "invert": True
    }
}


# --- Data Loading ---

def load_timeseries_json(json_path: Path) -> Dict[str, Any]:
    """Load time-series JSON from temporal_analyzer.py output."""
    with open(json_path) as f:
        return json.load(f)


def _to_float(val: Any) -> Optional[float]:
    """Best-effort conversion from JSON values to float (handles '9.55%', 'inf')."""
    try:
        if val is None:
            return None
        if isinstance(val, (int, float)):
            # guard against NaN/inf rendering issues; keep inf as None for plotting
            if val != val or val == float('inf') or val == float('-inf'):
                return None
            return float(val)
        if isinstance(val, str):
            s = val.strip()
            if s in {"∞", "inf", "-inf", "Infinity", "-Infinity"}:
                return None
            if s.endswith('%'):
                s = s[:-1]
            return float(s)
    except Exception:
        return None
    return None


def parse_timeseries_data(data: Dict[str, Any]) -> Tuple[List[datetime], Dict[str, List[float]]]:
    """
    Parse time-series data into plottable format.

    Returns:
        (dates, metrics_dict) where metrics_dict maps metric_name -> list of values
    """
    revisions = list(reversed(data.get('revisions', [])))  # oldest-first for chronological x-axis

    dates: List[datetime] = []
    metrics = {
        "propagation-cost": [],
        "m-score": [],
        "decoupling-level": [],
        "independence-level": []
    }

    for rev in revisions:
        # Parse date - handle both 'date' and 'commit_date' fields
        date_str = (rev.get('commit_date') or rev.get('date', '')).strip()
        # Normalize common forms: 'YYYY-MM-DD hh:mm:ss +0000' or 'YYYY-MM-DD'
        clean = date_str.split('+')[0].strip()
        dt = None
        try:
            dt = datetime.fromisoformat(clean.replace(' ', 'T'))
        except Exception:
            try:
                dt = datetime.strptime(clean.split()[0], '%Y-%m-%d')
            except Exception:
                dt = datetime.now()

        dates.append(dt)

        # Extract metrics
        rev_metrics = rev.get('metrics', {})
        for metric_name in metrics.keys():
            value = _to_float(rev_metrics.get(metric_name))
            metrics[metric_name].append(value)

    return dates, metrics


# --- Plotting Functions ---

def add_threshold_zones(ax, config: Dict[str, Any]):
    """Add colored background zones showing good/bad ranges."""
    thresholds = config['thresholds']
    colors = config['colors']

    for zone_name, (min_val, max_val) in thresholds.items():
        color = colors.get(zone_name, '#ffffff')
        ax.axhspan(min_val, max_val, alpha=0.2, color=color, zorder=0)


def annotate_significant_changes(ax, dates: List[datetime], values: List[float], threshold: float = 10.0):
    """Annotate points where metric changed by more than threshold%."""
    for i in range(1, len(values)):
        if values[i] is None or values[i-1] is None:
            continue

        delta = values[i] - values[i-1]
        abs_delta = abs(delta)
        if abs_delta >= threshold:
            direction = "↑" if delta > 0 else "↓"
            rel_pct = (delta / values[i-1] * 100.0) if values[i-1] != 0 else 0.0
            from_label = dates[i-1].strftime("%Y-%m")
            to_label = dates[i].strftime("%Y-%m")
            ax.annotate(
                f"{direction}{abs_delta:.1f} pts ({rel_pct:+.1f}%)\n{from_label}→{to_label}",
                xy=(dates[i], values[i]),
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=1)
            )


def plot_metric(dates: List[datetime], values: List[float], metric_name: str, ax=None, show_annotations: bool = True):
    """Plot a single metric with threshold zones and annotations."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    config = METRIC_CONFIG[metric_name]

    # Add threshold zones
    add_threshold_zones(ax, config)

    # Filter out None values
    valid_data = [(d, v) for d, v in zip(dates, values) if v is not None]
    if not valid_data:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        return

    plot_dates, plot_values = zip(*valid_data)

    # Plot line
    ax.plot(plot_dates, plot_values, marker='o', linewidth=2, markersize=6, label=config['title'])

    # Annotate significant changes
    if show_annotations:
        annotate_significant_changes(ax, dates, values, threshold=5.0)

    # Formatting
    ax.set_title(f"{config['title']}\n{config['description']}", fontsize=12, fontweight='bold')
    ax.set_xlabel("Commit Date", fontsize=10)
    ax.set_ylabel(config['ylabel'], fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())

    # Rotate date labels
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Set y-axis limits
    ax.set_ylim(0, 100)

    # Add legend for threshold zones
    from matplotlib.patches import Patch
    legend_elements = []
    for zone_name, color in config['colors'].items():
        legend_elements.append(Patch(facecolor=color, alpha=0.5, label=zone_name.capitalize()))

    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    plt.tight_layout()


def plot_combined_overview(dates: List[datetime], metrics_dict: Dict[str, List[float]], output_path: Optional[Path] = None):
    """Plot all four metrics in a 2x2 grid."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Architecture Metrics Evolution - Combined Overview', fontsize=16, fontweight='bold')

    metric_names = ["propagation-cost", "m-score", "decoupling-level", "independence-level"]

    for idx, metric_name in enumerate(metric_names):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]

        values = metrics_dict[metric_name]
        plot_metric(dates, values, metric_name, ax=ax, show_annotations=False)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")

    return fig


def plot_individual_metrics(dates: List[datetime], metrics_dict: Dict[str, List[float]], output_dir: Path):
    """Generate individual detailed plots for each metric."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric_name, values in metrics_dict.items():
        fig, ax = plt.subplots(figsize=(14, 8))
        plot_metric(dates, values, metric_name, ax=ax, show_annotations=True)

        output_path = output_dir / f"{metric_name}_timeseries.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
        plt.close(fig)


def calculate_summary_stats(values: List[float]) -> Dict[str, float]:
    """Calculate summary statistics for a metric."""
    valid_values = [v for v in values if v is not None]

    if not valid_values:
        return {
            'min': None,
            'max': None,
            'mean': None,
            'first': None,
            'last': None,
            'change': None,
            'volatility': None
        }

    import statistics

    first_val = valid_values[0]
    last_val = valid_values[-1]
    change = last_val - first_val
    volatility = statistics.stdev(valid_values) if len(valid_values) > 1 else 0

    return {
        'min': min(valid_values),
        'max': max(valid_values),
        'mean': statistics.mean(valid_values),
        'first': first_val,
        'last': last_val,
        'change': change,
        'volatility': volatility
    }


def print_summary_report(dates: List[datetime], metrics_dict: Dict[str, List[float]]):
    """Print text summary of trends and statistics."""
    print("\n" + "="*70)
    print("TEMPORAL ANALYSIS SUMMARY")
    print("="*70)

    if dates:
        print(f"\nTime Range: {dates[-1].date()} to {dates[0].date()}")
        print(f"Revisions: {len(dates)}")

    print("\n" + "-"*70)
    print("METRIC TRENDS")
    print("-"*70)

    for metric_name, values in metrics_dict.items():
        config = METRIC_CONFIG[metric_name]
        stats = calculate_summary_stats(values)

        print(f"\n{config['title']}:")
        print(f"  {config['description']}")

        if stats['first'] is not None:
            print(f"  First:  {stats['first']:.2f}%")
            print(f"  Last:   {stats['last']:.2f}%")
            print(f"  Change: {stats['change']:+.2f}% ({'improved' if stats['change'] > 0 and config['invert'] else 'degraded' if stats['change'] < 0 and config['invert'] else 'changed'})")
            print(f"  Mean:   {stats['mean']:.2f}%")
            print(f"  Range:  {stats['min']:.2f}% - {stats['max']:.2f}%")
            print(f"  StdDev: {stats['volatility']:.2f}%")

    print("\n" + "="*70 + "\n")


# --- Main API ---

def plot_timeseries(json_path: Path, output_dir: Optional[Path] = None, show: bool = False):
    """
    Main function to generate all plots from temporal analysis JSON.

    Args:
        json_path: Path to timeseries JSON from temporal_analyzer.py
        output_dir: Optional output directory for plots (defaults to same dir as JSON)
        show: Whether to display plots interactively
    """
    json_path = Path(json_path)

    if not json_path.exists():
        raise FileNotFoundError(f"Time-series JSON not found: {json_path}")

    # Load data
    print(f"\nLoading time-series data from: {json_path}")
    data = load_timeseries_json(json_path)
    dates, metrics_dict = parse_timeseries_data(data)

    repo_name = data.get('repo', 'unknown')
    print(f"Repository: {repo_name}")
    print(f"Revisions analyzed: {len(dates)}")

    # Setup output directory
    if output_dir is None:
        output_dir = json_path.parent / f"{repo_name}_plots"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Print summary
    print_summary_report(dates, metrics_dict)

    # Generate plots
    print("\nGenerating plots...")

    # Combined overview
    combined_path = output_dir / "metrics_overview.png"
    plot_combined_overview(dates, metrics_dict, combined_path)

    # Individual metrics
    plot_individual_metrics(dates, metrics_dict, output_dir)

    print(f"\nAll plots saved to: {output_dir}")

    # Show interactively if requested
    if show:
        plt.show()


# --- CLI ---

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate time-series plots from DV8 temporal analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot from temporal analysis output
  python3 metric_plotter.py --json pdfbox_timeseries.json

  # Specify output directory
  python3 metric_plotter.py --json data.json --output ./plots

  # Show plots interactively
  python3 metric_plotter.py --json data.json --show
        """
    )

    parser.add_argument(
        "--json",
        required=True,
        help="Path to time-series JSON from temporal_analyzer.py"
    )
    parser.add_argument(
        "--output",
        help="Output directory for plots (default: same dir as JSON)"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively"
    )

    args = parser.parse_args()

    try:
        plot_timeseries(
            json_path=Path(args.json),
            output_dir=Path(args.output) if args.output else None,
            show=args.show
        )
        return 0

    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
