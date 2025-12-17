from dataclasses import dataclass
from pathlib import Path

import matplotlib
from matplotlib.axes import Axes
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.pyplot import subplots, tight_layout

from numpy import cos, linspace, nanmean, nanstd, radians, sin
from pandas import DataFrame, crosstab, read_excel

from helpers.format_figure import format_axes

matplotlib.use("TkAgg")


@dataclass
class ImagingProbeFanBeamConfiguration:
    radius_of_curvature_in_meters: float
    imaging_depth_in_meters: float
    num_apertures: int
    beam_angle_in_degrees: float


def get_probe_fan_beam_polygon(probe_config: ImagingProbeFanBeamConfiguration) -> list[tuple[float, float]]:
    half_angle = probe_config.beam_angle_in_degrees / 2
    apertures_angles = linspace(-half_angle, half_angle, probe_config.num_apertures)
    aperture_y_coords_re_curvature_origin = probe_config.radius_of_curvature_in_meters * cos(radians(apertures_angles))
    aperture_x_coords_re_curvature_origin = probe_config.radius_of_curvature_in_meters * sin(radians(apertures_angles))

    a_line_tips_x = aperture_x_coords_re_curvature_origin + probe_config.imaging_depth_in_meters * sin(
        radians(apertures_angles)
    )
    a_line_tips_y = aperture_y_coords_re_curvature_origin + probe_config.imaging_depth_in_meters * cos(
        radians(apertures_angles)
    )

    image_origin_x = a_line_tips_x.mean()
    image_origin_y = probe_config.radius_of_curvature_in_meters

    aperture_points = [
        (x - image_origin_x, y - image_origin_y)
        for (x, y) in zip(aperture_x_coords_re_curvature_origin, aperture_y_coords_re_curvature_origin)
    ]

    a_line_tips = [(x - image_origin_x, y - image_origin_y) for (x, y) in zip(a_line_tips_x, a_line_tips_y)]

    return a_line_tips + aperture_points[::-1] + [a_line_tips[0]]


def plot_heatmap_metric(
    stats_df: DataFrame,
    metric_col: str,
    x_dim: str,
    y_dim: str,
    constant_dim: str,
    constant_value: float,
    ax,
    vmin=None,
    vmax=None,
    cmap="Blues",
    overlay_polygon: list | None = None,
    title: str | None = None,
    value_annotations: bool = True,
    plot_hor_roi_extent_in_m: float | None = 40e-3,
    print_results: bool = False
):
    """
    Plot a heatmap of a 3-D metric sliced into a 2-D plane.
    stats_df : DataFrame
        DataFrame indexed by (lateral, elevational, axial) positions, containing the metric to plot.
    metric_col : str
        Name of the column in stats_df to plot.
    x_dim, y_dim : str
        Names of the dimensions for the horizontal and vertical axes.
    constant_dim : str
        The dimension held constant to form the plane.
    constant_value : float
        The value of constant_dim to slice on.
    overlay_polygon: list | None
        A list of (x, y) points in physical coordinates (meters) to be scaled and plotted on top.
    """

    metric = stats_df[metric_col]

    # Ensure index is a DataFrame with columns we can query
    df = metric.reset_index()

    # Slice at the desired plane
    df_plane = df[df[constant_dim] == constant_value]

    # Pivot to 2-D table (y rows × x columns)
    plane = (
        df_plane.pivot(index=y_dim, columns=x_dim, values=metric.name).sort_index(axis=0).sort_index(axis=1)
    )
    roi_only = plane.loc[:, plane.columns <= plot_hor_roi_extent_in_m]

    if print_results:
        print(f"PLANE: {constant_value * 1e3} mm ----------")
        print(f"Mean in plane {metric_col} at {constant_value * 1e3} mm: {nanmean(plane.values) * 1e3}")
        print(f"STD in plane {metric_col} at {constant_value * 1e3} mm: {nanstd(plane.values) * 1e3}")
        print(f"Mean in ROI in plane {metric_col} at {constant_value * 1e3} mm:"
              f" {nanmean(roi_only.values) * 1e3}")
        print(f"STD in ROI in plane {metric_col} at {constant_value * 1e3} mm:"
              f" {nanstd(roi_only.values) * 1e3}")

    # Plot in millimetres
    sns.heatmap(
        plane * 1e3,
        ax=ax,
        annot=value_annotations,
        annot_kws={"size": 6},
        fmt=".2f",
        cmap=cmap,
        cbar=False,
        vmin=vmin,
        vmax=vmax,
        xticklabels=[f"{x*1e3:.0f}" for x in plane.columns],
        yticklabels=[f"{y*1e3:.0f}" for y in plane.index],
    )

    if overlay_polygon:
        # Get physical boundaries from the heatmap data (in meters)
        x_phys_min, x_phys_max = plane.columns.min(), plane.columns.max()
        y_phys_min, y_phys_max = plane.index.min(), plane.index.max()

        # Get index boundaries from the heatmap data
        x_idx_max = len(plane.columns) - 1
        y_idx_max = len(plane.index) - 1

        # Transform polygon coordinates from physical units to heatmap index units
        transformed_polygon = []
        for x_phys, y_phys in overlay_polygon:
            if x_phys is None or y_phys is None:
                continue
            # Linear interpolation to find index coordinate
            # Add 0.5 to center the point within the heatmap cell
            x_idx = 0.5 + ((x_phys - x_phys_min) / (x_phys_max - x_phys_min)) * x_idx_max
            # For y, the heatmap index is inverted relative to physical coordinates.
            y_idx = 0.5 + ((y_phys - y_phys_min) / (y_phys_max - y_phys_min)) * y_idx_max
            transformed_polygon.append((x_idx, y_idx))

        ROI_polygon = [
            (-20e-3, 10e-3),
            (-20e-3, 139e-3),
            (plot_hor_roi_extent_in_m, 139e-3),
            (plot_hor_roi_extent_in_m, 10e-3),
            (-20e-3, 10e-3)
        ]
        transformed_ROI_polygon = []
        for x_phys, y_phys in ROI_polygon:
            ROI_x_idx = 1.0 + ((x_phys - x_phys_min) / (x_phys_max - x_phys_min)) * x_idx_max
            ROI_y_idx = 1.0 + ((y_phys - y_phys_min) / (y_phys_max - y_phys_min)) * y_idx_max
            transformed_ROI_polygon.append((ROI_x_idx, ROI_y_idx))

        # Plot the transformed polygon
        poly_x, poly_y = zip(*transformed_polygon)
        ax.plot(poly_x, poly_y, color='black', linestyle='--', zorder=10)
        roi_poly_x, roi_poly_y = zip(*transformed_ROI_polygon)
        ax.plot(roi_poly_x, roi_poly_y, color='black', linestyle=':', zorder=11)

    ax.set_xlabel(f"{x_dim.replace('_position_motor','').title()} (mm)")
    ax.set_ylabel(f"{y_dim.replace('_position_motor','').title()} (mm)")
    if title is not None:
        ax.set_title(title)
    ax.set_aspect("equal")


def plot_results_at_elevation(
    frame: "DataFrame",
    error_axes: Axes,
    repeatability_axes: Axes,
    elevation: float,
    plot_error_hor_axis: bool = True,
    plot_error_ver_axis: bool = True,
    plot_rep_hor_axis: bool = True,
    plot_rep_ver_axis: bool = True,
    hor_roi_extent_in_m: float | None = 40e-3,
    print_results: bool = False
):
    """
    Plot mean error (bias magnitude) and repeatability magnitude at a specific elevation plane.
    """

    # --- Probe configuration for overlay ---
    probe_configuration = ImagingProbeFanBeamConfiguration(
        radius_of_curvature_in_meters=51.25e-3,
        imaging_depth_in_meters=150e-3,
        num_apertures=128,
        beam_angle_in_degrees=65,
    )
    fan_beam_polygon = get_probe_fan_beam_polygon(probe_configuration)

    # --- Plot Bias Magnitude ---
    plot_heatmap_metric(
        stats_df=frame,
        metric_col="bias_magnitude",
        x_dim="lateral_position_motor",
        y_dim="axial_position_motor",
        constant_dim="elevational_position_motor",
        constant_value=elevation,
        ax=error_axes,
        overlay_polygon=fan_beam_polygon,
        value_annotations=False,
        vmin=0.0,
        vmax=9.0,
        plot_hor_roi_extent_in_m=hor_roi_extent_in_m,
        print_results=print_results
    )
    error_axes.yaxis.set_visible(plot_error_ver_axis)
    error_axes.xaxis.set_visible(plot_error_hor_axis)

    # --- Plot Repeatability Magnitude ---
    plot_heatmap_metric(
        stats_df=frame,
        metric_col="repeatability_magnitude",
        x_dim="lateral_position_motor",
        y_dim="axial_position_motor",
        constant_dim="elevational_position_motor",
        constant_value=elevation,
        ax=repeatability_axes,
        overlay_polygon=fan_beam_polygon,
        value_annotations=False,
        vmin=0.0,
        vmax=5.0,
        plot_hor_roi_extent_in_m=hor_roi_extent_in_m,
        print_results=print_results
    )
    repeatability_axes.yaxis.set_visible(plot_rep_ver_axis)
    repeatability_axes.xaxis.set_visible(plot_rep_hor_axis)


def plot_likert_summary(data_path_1: Path):
    """
    Reads survey data, categorizes Likert-scale questions, and generates
    a summarized diverging stacked bar chart of participant satisfaction.

    Args:
        data_path_1 (str): File path for the first survey data CSV.
    """
    try:
        df = read_excel(data_path_1, nrows=12, usecols="B:R")
    except FileNotFoundError as e:
        print(f"Error: One of the files was not found. {e}")
        return

    response_levels = [
        "Strongly disagree", "Disagree", "Somewhat disagree", "Neutral",
        "Somewhat agree", "Agree", "Strongly agree"
    ]
    response_levels_lower = [r.lower() for r in response_levels]

    likert_questions = []
    for col in df.columns:
        if df[col].dtype == 'object':
            unique_answers = set(df[col].dropna().str.lower().unique())
            if unique_answers and unique_answers.issubset(set(response_levels_lower)):
                likert_questions.append(col)

    category_map = {
        'Hardware': [
            'The trackable stylet fit',
            'It was as easy to withdraw',
            'The fibre-optic cable between the trackable stylet and the tracking system did not affect',
            'the right length',
            'The presence of the tracking array'
        ],
        'Imaging': [
            'suitable image quality',
            'noticeable latency on the imaging feed'
        ],
        'Tracking Visualisation': [
            'clearly visible',
            'cursor did not obstruct',
            'interpret the distance'
        ],
        'Tracking Performance': [
            'latency of the displayed tracked',
            'update rate',
            'depth of the needle tip',
            'insertion angle of the needle',
            'tracking was accurate enough'
        ],
        'Overall Impact': [
            'presence of the tracking visualisation made it easier'
        ]
    }

    def get_category(question_title):
        q_lower = question_title.lower()
        for category, keywords in category_map.items():
            for keyword in keywords:
                if keyword.lower() in q_lower:
                    return category
        raise ValueError(f"Could not find category for {question_title}")

    # Melt the dataframe to long format to pool responses
    df_long = df[likert_questions].melt(var_name='question', value_name='response')
    df_long = df_long.dropna(subset=['response'])
    df_long['response'] = df_long['response'].str.lower()
    df_long['category'] = df_long['question'].map(get_category)

    category_counts = crosstab(df_long['category'], df_long['response'])

    print(df_long['question'].unique())

    summary_by_category = category_counts.apply(lambda x: x / x.sum() * 100, axis=1)

    summary_by_category = summary_by_category.reindex(columns=response_levels_lower, fill_value=0)
    summary_by_category = summary_by_category.reindex(list(category_map.keys()))

    # --- PLOTTING --- #
    fig, ax = subplots()

    colors = {
        'strongly disagree': '#d7191c', 'disagree': '#fdae61', 'somewhat disagree': '#fee08b',
        'neutral': 'lightgrey',
        'somewhat agree': '#d9ef8b', 'agree': '#91cf60', 'strongly agree': '#1a9850'
    }

    left_cols = ['strongly disagree', 'disagree', 'somewhat disagree']

    # Calculate the midpoint to center the neutral category at 0
    midpoint = summary_by_category[left_cols].sum(axis=1) + summary_by_category['neutral'] / 2

    # Plot each segment of the bars
    cumulative_sum = summary_by_category.cumsum(axis=1)
    for col in response_levels_lower:
        width = summary_by_category[col]
        start = cumulative_sum[col] - width - midpoint
        ax.barh(summary_by_category.index, width, left=start, color=colors[col], label=col.title())

    ax.axvline(0, color='grey', linewidth=0.8, linestyle='--')
    ax.plot(
        [0, 0], [-0.12, 0], transform=ax.get_xaxis_transform(),
        color='grey', linewidth=0.8, linestyle='--', clip_on=False, zorder=1
    )
    ax.set_xlabel('')
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.tick_params(axis='x', which='both', bottom=False, top=False)

    for i, category in enumerate(summary_by_category.index):
        cumulative = -midpoint[category]
        for col in response_levels_lower:
            value = summary_by_category.loc[category, col]
            if round(value) >= 5:
                label = f"{round(value):d}\\%"
                xpos = cumulative + value / 2
                ypos = i + 0.04

                num_digits = len(str(round(value)))
                text_shift = 0.01 * num_digits * (1 if xpos >= 0 else -1)
                xpos -= text_shift

                ax.text(
                    xpos, ypos, label,
                    ha='center', va='center', color='black'
                )
            cumulative += value

    ax.text(
        -1, -0.05,
        '← Disagreement',
        ha='right', va='top',  # fontsize=9,
        transform=ax.get_xaxis_transform()
    )

    ax.text(
        1.5, -0.05,
        'Agreement →',
        ha='left', va='top',  # fontsize=9,
        transform=ax.get_xaxis_transform()
    )

    ax.invert_yaxis()

    handles, labels = ax.get_legend_handles_labels()
    unique_labels_map = dict(zip(labels, handles))

    desired_legend_order_titles = [
        response_levels[0].title(),
        response_levels[1].title(),
        response_levels[2].title(),
        response_levels[3].title(),
        '',
        '',
        response_levels[4].title(),
        response_levels[5].title(),
        response_levels[6].title()
    ]

    ordered_handles = []
    ordered_labels = []
    for label_title in desired_legend_order_titles:
        if label_title:  # If not an empty string, get the actual handle
            ordered_labels.append(label_title)
            ordered_handles.append(unique_labels_map.get(label_title, Line2D([0], [0], color='none')))
        else:  # For empty string placeholders, create an invisible handle
            ordered_labels.append('')
            ordered_handles.append(Line2D([0], [0], color='none'))  # Invisible handle

    ax.legend(ordered_handles, ordered_labels, loc='upper center', bbox_to_anchor=(0.5, 1.35), ncol=3, frameon=False)
    format_axes(ax)
    tight_layout(rect=[0, 0.05, 1, 0.95])
