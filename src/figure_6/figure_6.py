from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from helpers.format_figure import format_axes, latexify


def parse_mean_std(series):
    return series.str.split(' ± ', expand=True).astype(float)


if __name__ == "__main__":

    SAVE = False

    NUM_STDS_TO_PLOT = 2

    file_path = Path(__file__).parent / Path('xct_summary_results.csv')
    data = pd.read_csv(file_path, header=1)

    # Rename columns for easier access
    data.columns = ['xct_lateral', 'xct_elevational', 'xct_axial', 'disp_lateral', 'disp_elevational', 'disp_axial', 'disp_distance']

    # Parse the displacement columns
    for col in ['disp_lateral', 'disp_elevational', 'disp_axial', 'disp_distance']:
        data[[f'{col}_mean', f'{col}_std']] = parse_mean_std(data[col])

    # Recalculate the tracked locations
    tracked_lateral = data['xct_lateral'] + data['disp_lateral_mean']
    tracked_elevational = data['xct_elevational'] + data['disp_elevational_mean']
    tracked_axial = data['xct_axial'] + data['disp_axial_mean']

    # Calculate and print the average displacement and standard deviation
    mean_displacement = data['disp_distance_mean'].mean()
    std_displacement = data['disp_distance_mean'].std()
    print(data[['xct_axial', 'disp_distance_mean', 'disp_distance_std']])
    print(f'Average displacement: {mean_displacement:.2f} ± {std_displacement:.2f} mm')

    # --- Axial-Lateral Plot ---
    latexify(fig_width_column_fraction=0.45, fig_height_column_fraction=0.5)
    fig2, ax2 = plt.subplots(1, 1)
    ax2.scatter(data['xct_lateral'], data['xct_axial'], edgecolors='gray', marker='o', facecolors='None', label='XCT Location')
    ax2.errorbar(tracked_lateral, tracked_axial,
                 xerr=data['disp_lateral_std'] * NUM_STDS_TO_PLOT,
                 yerr=data['disp_axial_std'] * NUM_STDS_TO_PLOT,
                 fmt='x', color='k', ecolor='k', elinewidth=2, capsize=2, label='Tracked Location')
    ax2.set_xlabel('Lateral Pos. (mm)')
    ax2.set_ylabel('Axial Pos. (mm)')
    ax2.grid(True)
    ax2.set_aspect('equal', adjustable='box')
    ax2.yaxis.set_inverted(True)
    format_axes(ax2)
    plt.tight_layout()
    if SAVE:
        plt.savefig('xct_results_plot_axial_lateral.pdf', bbox_inches='tight')

    # --- Axial-Elevational Plot ---
    latexify(fig_width_column_fraction=0.3, fig_height_column_fraction=0.5)
    fig3, ax3 = plt.subplots(1, 1)
    ax3.scatter(data['xct_elevational'], data['xct_axial'],  edgecolors='gray', marker='o', facecolors='None', label='XCT Location')
    ax3.errorbar(tracked_elevational, tracked_axial,
                 xerr=data['disp_elevational_std'] * NUM_STDS_TO_PLOT,
                 yerr=data['disp_axial_std'] * NUM_STDS_TO_PLOT,
                 fmt='x', color='k', ecolor='k', elinewidth=2, capsize=2, label='Tracked Location')
    ax3.set_xlabel('Elev. Pos. (mm)')
    ax3.set_ylabel('Axial Pos. (mm)')
    ax3.grid(True)
    ax3.set_xlim([-15, 15])
    ax3.set_aspect('equal', adjustable='box')
    ax3.yaxis.set_inverted(True)
    format_axes(ax3)
    plt.tight_layout()
    if SAVE:
        plt.savefig('xct_results_plot_axial_elevational.pdf', bbox_inches='tight')

    # --- Legend Figure ---
    latexify(fig_width_column_fraction=0.58)
    fig_legend = plt.figure()
    handles, labels = ax2.get_legend_handles_labels()
    fig_legend.legend(handles, labels, loc='center', frameon=False, ncol=2)
    plt.axis('off')
    if SAVE:
        plt.savefig('xct_results_plot_legend.pdf', bbox_inches='tight')

    plt.show()
