import numpy as np
import matplotlib.pyplot as plt
import utils


def plot_series(file_paths, bounds, cmap="RdBu_r", dsets=["CMI"], title_format="time"):
    metas = [utils.parse_goes_filename(f) for f in file_paths for d in dsets]
    image_list = [
        utils.warp_subset(f, bounds=bounds, dset=d) for f in file_paths for d in dsets
    ]

    nfiles = len(image_list)
    if nfiles > 3:
        ntiles = int(np.ceil(np.sqrt(nfiles)))
        layout = (ntiles, ntiles)
    else:
        layout = (1, nfiles)

    fig, axes = plt.subplots(*layout, sharex=True, sharey=True, squeeze=False)
    for ax, cm, m in zip(axes.ravel(), image_list, metas):
        axim = ax.imshow(cm, cmap=cmap)
        fig.colorbar(axim, ax=ax)
        if title_format == "time":
            title = m["start_time"].strftime("%H:%M")
        elif title_format == "channel":
            title = m["channel"]
        else:
            raise ValueError("Unknown title_format")
        ax.set_title(title)

    fig.suptitle(m["start_time"].strftime("%Y-%m-%d"))
    fig.tight_layout()