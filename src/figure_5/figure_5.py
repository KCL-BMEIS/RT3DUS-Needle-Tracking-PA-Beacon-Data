from src.helpers import plot_results_at_elevation
from matplotlib.pyplot import savefig, show, subplots, tight_layout
from pandas import DataFrame, concat, read_csv
import numpy as np

from helpers.format_figure import format_axes, latexify


def vector_magnitude(df: DataFrame, components: list[str]):
    """Compute the Euclidean magnitude for given component columns."""
    return np.sqrt((df[components] ** 2.0).sum(axis=1))


if __name__ == "__main__":

    SAVE_FIG = False

    STATS_FILE_NAMES = [
        "water_accuracy_1.csv",
        "water_accuracy_2.csv"
    ]

    elevations = [-0.00, -0.01, -0.02]

    latexify(fig_width_column_fraction=0.9, fig_height_column_fraction=0.6)

    all_scans = []
    for file_name in STATS_FILE_NAMES:
        df = read_csv(
            file_name,
            index_col=["lateral_position_motor", "elevational_position_motor", "axial_position_motor"],
        )
        all_scans.append(df)
    concat_frames = concat(all_scans)
    avg_frame = concat_frames.groupby(level=list(range(concat_frames.index.nlevels))).mean()
    std_frame = concat_frames.groupby(level=list(range(concat_frames.index.nlevels))).std()

    print(f"Tracking accuracy assessed at {len(avg_frame)} locations")

    avg_frame["bias_magnitude"] = vector_magnitude(
        avg_frame, [
            "mean_error_lateral_position_tracked",
            "mean_error_elevational_position_tracked",
            "mean_error_axial_position_tracked",
        ]
    )

    avg_frame["repeatability_magnitude"] = vector_magnitude(
        avg_frame, [
            "repeatability_lateral_position_tracked",
            "repeatability_elevational_position_tracked",
            "repeatability_axial_position_tracked",
        ]
    )

    std_frame["bias_magnitude"] = vector_magnitude(
        std_frame, [
            "mean_error_lateral_position_tracked",
            "mean_error_elevational_position_tracked",
            "mean_error_axial_position_tracked",
        ]
    )

    std_frame["repeatability_magnitude"] = vector_magnitude(
        std_frame, [
            "repeatability_lateral_position_tracked",
            "repeatability_elevational_position_tracked",
            "repeatability_axial_position_tracked",
        ]
    )

    # Whole-volume mean
    mean_bias_mag = avg_frame["bias_magnitude"].mean()
    std_bias_mag = avg_frame["bias_magnitude"].std()
    mean_repeat_mag = avg_frame["repeatability_magnitude"].mean()
    std_repeat_mag = avg_frame["repeatability_magnitude"].std()

    print(f"=== Mean over whole volume ===")
    print(f"Bias magnitude: {mean_bias_mag * 1e3:.3f} ± {std_bias_mag * 1e3:.3f} mm")
    print(f"Repeatability magnitude: {mean_repeat_mag * 1e3:.3f} ± {std_repeat_mag * 1e3:.3f} mm")

    # ROI restriction
    roi_volume = avg_frame.query('lateral_position_motor <= 0.04')
    mean_bias_mag_roi = roi_volume["bias_magnitude"].mean()
    std_bias_mag_roi = roi_volume["bias_magnitude"].std()
    mean_repeat_mag_roi = roi_volume["repeatability_magnitude"].mean()
    std_repeat_mag_roi = roi_volume["repeatability_magnitude"].std()

    print(f"=== Mean over ROI (lateral ≤ 40 mm) ===")
    print(f"Bias magnitude: {mean_bias_mag_roi * 1e3:.3f} ± {std_bias_mag_roi * 1e3:.3f} mm")
    print(f"Repeatability magnitude: {mean_repeat_mag_roi * 1e3:.3f} ± {std_repeat_mag_roi * 1e3:.3f} mm")

    # ------------------------------------------------------------
    # Per-plane results and plotting
    # ------------------------------------------------------------
    print("\n=== Per-elevation results ===")

    fig2, axes = subplots(2, 3)
    for elev_index, elevation in enumerate(elevations):
        print(f"\n--- Elevation {elevation * 1e3:.1f} mm ---")

        plot_results_at_elevation(
            frame=avg_frame,
            error_axes=axes[0, elev_index],
            repeatability_axes=axes[1, elev_index],
            elevation=elevation,
            plot_error_ver_axis=elev_index == 0,
            plot_error_hor_axis=False,
            plot_rep_ver_axis=elev_index == 0,
            plot_rep_hor_axis=True,
            hor_roi_extent_in_m=40e-3,
            print_results=True
        )

    # Colorbars
    cax_top = fig2.add_axes([0.90, 0.567, 0.02, 0.35])
    cax_bottom = fig2.add_axes([0.90, 0.167, 0.02, 0.35])
    fig2.colorbar(axes.flatten()[0].collections[0], cax=cax_top, label='Error (mm)')
    fig2.colorbar(axes.flatten()[3].collections[0], cax=cax_bottom, label='Repeatability (mm)')

    # Titles and formatting
    for elev, ax in zip(elevations, axes[0, :]):
        ax.set_title(f"{abs(elev) * 1e3:.0f} mm Elevation")
    for ax in axes.flatten():
        format_axes(ax)

    tight_layout(rect=[0, 0, 0.9, 1])

    if SAVE_FIG:
        savefig('water_results_plot.pdf', bbox_inches='tight')

    fig2.suptitle("Figure 5")
    tight_layout(rect=[0, 0, 0.9, 1])

    # ------------------------------------------------------------
    # Checking the standard deviation across scans
    # ------------------------------------------------------------

    fig2, axes = subplots(2, 3)
    for elev_index, elevation in enumerate(elevations):
        plot_results_at_elevation(
            frame=std_frame,
            error_axes=axes[0, elev_index],
            repeatability_axes=axes[1, elev_index],
            elevation=elevation,
            plot_error_ver_axis=elev_index == 0,
            plot_error_hor_axis=False,
            plot_rep_ver_axis=elev_index == 0,
            plot_rep_hor_axis=True,
            hor_roi_extent_in_m=40e-3,
            print_results=False
        )

    cax_top = fig2.add_axes([0.90, 0.567, 0.02, 0.35])
    cax_bottom = fig2.add_axes([0.90, 0.167, 0.02, 0.35])
    fig2.colorbar(axes.flatten()[0].collections[0], cax=cax_top, label='Error (mm)')
    fig2.colorbar(axes.flatten()[3].collections[0], cax=cax_bottom, label='Repeatability (mm)')

    for elev, ax in zip(elevations, axes[0, :]):
        ax.set_title(f"{abs(elev) * 1e3:.0f} mm Elevation")
    for ax in axes.flatten():
        format_axes(ax)

    fig2.suptitle("Standard deviation across scans")

    tight_layout(rect=[0, 0, 0.9, 1])

    show()
