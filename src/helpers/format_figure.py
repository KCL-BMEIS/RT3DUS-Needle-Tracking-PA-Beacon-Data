import warnings
warnings.filterwarnings("ignore")

import os
import re
from pathlib import Path

import matplotlib
from numpy import sqrt
from cycler import cycler
import seaborn as sns

os.environ['PATH'] += os.pathsep + "/Library/TeX/texbin"

LATEX_CLASS_FILE_PATH = Path(__file__).parent / Path('iopjournal.cls')


def get_column_width_mm_from_latex_class(file_path: Path):
    with open(file_path, 'r') as f:
        content = f.read()
    match = re.search(r'\\setlength\\textwidth{(\d+\.?\d*)mm}', content)
    if match:
        return float(match.group(1))
    return None


def get_font_from_latex_class(file_path: Path):
    with open(file_path, 'r') as f:
        content = f.read()
    match = re.search(r'\\renewcommand<...>*\{\s*\\fontfamily\{(\w+)\}', content)
    if match:
        return match.group(1)
    return 'serif'


def get_caption_font_size_from_latex_class(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    match = re.search(r'\\renewcommand\{\\@makecaption\}\[2\].*\\sbox\\@tempboxa\{\\fontsize\{(\d+)\}', content, re.DOTALL)
    if match:
        return int(match.group(1))
    print(f"Failed to find font size in class file")
    return 10



COLUMN_WIDTH_MM = get_column_width_mm_from_latex_class(LATEX_CLASS_FILE_PATH)
COLUMN_WIDTH_INCHES = COLUMN_WIDTH_MM / 25.4
MAX_HEIGHT_INCHES = 15.0
FONT_SIZE = get_caption_font_size_from_latex_class(LATEX_CLASS_FILE_PATH)
FONT_FAMILY = get_font_from_latex_class(LATEX_CLASS_FILE_PATH)

print(f"Using font size {FONT_SIZE} and family {FONT_FAMILY}")


def latexify(fig_width_column_fraction, fig_height_column_fraction=None):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.
    Parameters
    """
    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    font_size = FONT_SIZE

    fig_width = fig_width_column_fraction * COLUMN_WIDTH_INCHES
    if fig_height_column_fraction is None:
        golden_mean = (sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
        fig_height = fig_width * golden_mean  # height in inches
    else:
        fig_height = fig_height_column_fraction * COLUMN_WIDTH_INCHES

    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large: " + str(fig_height) +
              " so will reduce to " + str(MAX_HEIGHT_INCHES) + " inches.")
        fig_height = MAX_HEIGHT_INCHES

    color_cycle = sns.color_palette(palette='muted')

    preamble = [
        r'\usepackage{gensymb}',
        r'\usepackage{textgreek}',
        r'\usepackage{siunitx}'
    ]

    params = {#'backend': 'ps',
              'text.latex.preamble': '\n'.join(preamble),
              'axes.labelsize': font_size,
              'axes.titlesize': font_size,
              'axes.prop_cycle': cycler(color=color_cycle),
              'font.size': font_size,
              'legend.fontsize': font_size,
              'xtick.labelsize': font_size,
              'ytick.labelsize': font_size,
              'text.usetex': True,
              'figure.figsize': [fig_width, fig_height],
              'font.family': FONT_FAMILY
              }
    matplotlib.rcParams.update(params)


def format_axes(ax, twinx=False):

    if twinx:
        yaxis_side = 'right'
        empty_side = 'left'
    else:
        yaxis_side = 'left'
        empty_side = 'right'

    for child in ax.get_children():
        if isinstance(child, matplotlib.lines.Line2D):
            child.set_linewidth(0.75)
    SPINE_COLOR = 'gray'
    for spine in ['top', empty_side]:
        ax.spines[spine].set_visible(False)
    for spine in [yaxis_side, 'bottom']:
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.5)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position(yaxis_side)
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction='out', color=SPINE_COLOR)
    return ax