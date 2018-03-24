#!/usr/bin/env python

from __future__ import print_function

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pandas as pd
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file",
        nargs="+",
        help="CSV files to plot",
    )
    parser.add_argument(
        "-i",
        dest="interactive",
        help="Shows plot interactively",
        action="store_true",
        default=False,
    )
    return parser.parse_args()


def plot(f, args):
    try:
        df = pd.read_csv(f)
        print("Plotting file: {}".format(f))
    except pd.parser.CParserError:
        print("Failed to load file: {}".format(f))
        return

    tensor_rank = len(df.columns) - 1

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    if tensor_rank == 2:
        df_pivot = df.pivot(index="dimension_1", columns="dimension_2", values="value")

        # value histogram
        raw_values = df_pivot.values.flatten()
        axes[0].hist(raw_values, bins=70)

        # heatmap
        im = axes[1].imshow(df_pivot, aspect="auto", interpolation="none")
        divider = make_axes_locatable(axes[1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax)

        fig.suptitle(f, fontsize=20)
        if args.interactive:
            plt.show()
        fig.savefig(f + ".png")

    else:
        print("Tensor of rank {} are not supported yet.".format(tensor_rank))


if __name__ == "__main__":
    args = parse_args()

    for f in args.file:
        plot(f, args)
