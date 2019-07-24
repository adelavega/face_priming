import matplotlib.pyplot as plt
import pandas as pd

def standardize(i):
    """ Standardize numpy image """
    i = i - i.min()
    i = (i / i.max())
    return i

def plot_clusters(clusters, figsize=(10, 45), max_rows=None):
    width, height = pd.DataFrame(clusters).shape
    if max_rows is not None and height > max_rows:
        height = max_rows

    f, axarr = plt.subplots(height, width, figsize=figsize)
    [axi.set_axis_off() for axi in axarr.ravel()]
    for ix, images in enumerate(clusters):
        for iy, i in enumerate(images):
            if iy < max_rows:
                axarr[iy, ix].imshow(standardize(images[iy][1]))