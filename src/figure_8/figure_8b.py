from pathlib import Path

import seaborn as sns
from matplotlib.pyplot import savefig, show

from helpers import plot_likert_summary
from helpers.format_figure import latexify

SAVE = False

if __name__ == '__main__':
    FIG_WIDTH = 0.9
    sns.set_style("whitegrid")
    latexify(fig_width_column_fraction=FIG_WIDTH)
    file1 = Path(__file__).parent / Path("usability_survey.xlsx")
    plot_likert_summary(file1)
    if SAVE:
        savefig(f'survey_results_w{FIG_WIDTH}.pdf', bbox_inches='tight')
    show()
