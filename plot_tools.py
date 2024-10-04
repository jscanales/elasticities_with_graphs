"""
This module defines some functions and constants that are common to plots_application and plots_simulations
"""

import os
from estimation import ModelInputParams, IVOptions
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# VERSION = "v108"
VERSION = "vClean"

def save_figure(
    fig:Figure, figname: str, version: str = VERSION, fmt: str = "pdf", root:str = "simulations/EXP", replace_existing_file:bool = True
):
    name = f"./plots/{root}_{version}_{figname}.{fmt}"
    if os.path.isfile(name) and not replace_existing_file:
        raise Exception("!!!\nCareful! Change the version or delete the existing file\n!!!\nVERSION will also affect the output file name")
    else:
        fig.savefig(name, bbox_inches='tight')

def numerate_subplots(axss, x:float=0, y:float=1.15):
    labels: list[str] = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o']
    for i, hax in enumerate(axss.flat):
        label: str = labels[i]
        hax.text(x, y, f'{label})', transform=hax.transAxes, va='top', ha='left', size='large')

def make_legend_arrow(legend, orig_handle,
                      xdescent, ydescent,
                      width, height, fontsize):
    p = mpatches.FancyArrow(0, 0.5*height, width, 0, width=1, length_includes_head=True, head_width=5, head_length=5)
    return p


def empty_handles(n:int) -> list:
    return [mpatches.Rectangle((0,0), 1, 1, fill=False, edgecolor='none',
                                visible=False) for i in range(n)]


def func_replace_ellipsis(string:str) -> str:
        return string.replace(",...,", ":").replace(", ...,", ":")

"Plotting constants"
REGULAR_WIDTH = 6
HEIGHT = REGULAR_WIDTH*(6/10)
WIDE_WIDTH = REGULAR_WIDTH*(3/2)
EXTRA_WIDTH = REGULAR_WIDTH*2

"Core estimators"
MODELS=[
        ModelInputParams(IVOptions.CONDITIONAL_WIND, order=26),
        ModelInputParams(IVOptions.REGULAR, order=26),
        ModelInputParams(IVOptions.CONDITIONAL_DEMAND, order=2, order_w=26),
        ModelInputParams(IVOptions.TRUNCATED_NUISANCE_ORDER, order=26, order_d=1),
        ModelInputParams(IVOptions.CLEAN_2dim_ORDER, order=26, order_price=1),
        ModelInputParams(IVOptions.TRUNCATED_IV_2dim_ORDER, order=26, order_price=1),
        ModelInputParams(IVOptions.IV_2dim_ORDER, order=26, order_price=1),
        ModelInputParams(IVOptions.CONDITIONAL_H0, order=26),
        ]

def plot_table(fig:Figure):
    cross = "$\\times$"
    check = "$\\checkmark$"
    table_data = [
        ["I", "II", "III"],        

        [cross, check, cross],

        [check, check, check],
        [check, cross, cross],

        [check, check, check],

        [check, check, cross],
        [cross, check, check],
        [check, cross, check],
        [check, check, check],
    ]

    table_pos = [1, 0.001, 0.2, 1.125]  # [left, bottom, width, height]

    table = plt.table(bbox=table_pos, cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.5, 1.5)  # Adjust the scaling if needed

    fig.patches.extend([table]) # type: ignore
