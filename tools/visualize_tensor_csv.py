#!/usr/bin/env python

from __future__ import print_function

import argparse
import sys

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
except ImportError:
    print("Failed to import matplotlib. This tool requires matplotlib.")

try:
    import pandas as pd
except ImportError:
    print("Failed to import pandas. This tool requires pandas.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tool to visualize tensors generated from Arraymancer's "
                    "Tensor.to_csv(...). It plots each given CSV file into a "
                    "corresponding PNG file with the same file name.")
    parser.add_argument(
        "file",
        nargs="+",
        help="CSV file(s) to plot",
    )
    parser.add_argument(
        "-i",
        dest="interactive",
        help="Shows plot interactively",
        action="store_true",
        default=False,
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def plot(f, args):
    try:
        df = pd.read_csv(f)
        print("\n *** Plotting file '{}'. Tensor value stats:\n{}".format(
            f, df["value"].describe())
        )
    except pd.parser.CParserError:
        print("Failed to load file: {}".format(f))
        return

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # value histogram
    raw_values = df["value"].values
    axes[0].hist(raw_values, bins=70)

    # rank specific plot: plain value plot (1D) or heatmap (2D)
    tensor_rank = len(df.columns) - 1
    if tensor_rank == 1:
        x_values = range(len(df))
        y_values = df["value"].values
        axes[1].plot(x_values, y_values, "o", ms=2)
    elif tensor_rank == 2:
        df_pivot = df.pivot(index="dimension_1", columns="dimension_2", values="value")
        im = axes[1].imshow(df_pivot, aspect="auto", interpolation="none")
        divider = make_axes_locatable(axes[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)
    else:
        axes[1].text(
            0.5, 0.5,
            "No visualization available for tensors of rank {}".format(tensor_rank),
            horizontalalignment="center", verticalalignment="center", fontsize=10,
            transform=axes[1].transAxes
        )

    tensor_shape = [df.iloc[:, i].max() + 1 for i in range(tensor_rank)]
    fig.suptitle("{} (shape: {})".format(f, tensor_shape), fontsize=16)
    if args.interactive:
        plt.show()
    fig.savefig(f + ".png")
    plt.close(fig)


if __name__ == "__main__":
    args = parse_args()

    for f in args.file:
        plot(f, args)
