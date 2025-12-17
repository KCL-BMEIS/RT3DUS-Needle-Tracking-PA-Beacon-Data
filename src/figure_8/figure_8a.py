import matplotlib.pyplot as plt

from helpers.format_figure import format_axes, latexify

SAVE = False

if __name__ == "__main__":
    FIG_WIDTH = 0.75

    latexify(fig_width_column_fraction=FIG_WIDTH, fig_height_column_fraction=0.25)

    # Data derived from the contingency table
    categories = [
        "All\nsucceeded",
        "Only failed\nuntracked",
        "Only failed\ntracked",
        "Failed\nboth"
    ]
    values = [6, 3, 1, 2]

    # Create the bar chart
    plt.figure()
    bars = plt.bar(categories, values, color='#d9ef8b', width=0.6)

    # Add labels
    plt.ylabel("Num. Participants")
    plt.gca().set_yticks([0, 2, 4, 6])

    format_axes(plt.gca())

    plt.tight_layout()

    if SAVE:
        plt.savefig(f'survey_contingency_{FIG_WIDTH}.pdf', bbox_inches='tight')

    plt.show()
