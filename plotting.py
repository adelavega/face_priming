import matplotlib.pyplot as plt
import pandas as pd
from random import sample
import numpy as np

def standardize(i):
    """ Standardize numpy image """
    i = i - i.min()
    i = (i / i.max())
    return i

def plot_clusters(clusters, figsize=(10, 45), n_sample=100, outfile=None):
    if n_sample is not None:
        clusters = [sample(c, n_sample) if len (c) > n_sample else c for c in clusters]
    width, height = pd.DataFrame(clusters).shape

    f, axarr = plt.subplots(height, width, figsize=figsize)
    [axi.set_axis_off() for axi in axarr.ravel()]
    for ix, images in enumerate(clusters):
        for iy, i in enumerate(images):
            if iy < height:
                axarr[iy, ix].imshow(standardize(images[iy][1]))

    if outfile is not None:
        plt.savefig(outfile)
        return outfile
    else:
        return f, axarr


def plot_sklearn_clusters(clusters, images, figsize=(10, 45), n_sample=50,
                          sort=True, outfile=None):
    unique_counts = np.unique(clusters.labels_[clusters.labels_!=-1],
                              return_counts=True)

    if sort is True:
        unique_labels = [x for x,_ in
                  sorted(zip(*unique_counts), reverse=True, key=lambda p: p[1])]
    else:
        unique_labels = unique_counts[0]

    unique_labels = np.array([-1] + unique_labels)

    f, axarr = plt.subplots(n_sample, len(unique_labels), figsize=figsize)
    [axi.set_axis_off() for axi in axarr.ravel()]

    for ix, i in enumerate(unique_labels):
        clust_ix = np.where(clusters.labels_ == i)[0]

        if n_sample is not None and len(clust_ix) > n_sample:
            clust_ix = np.random.choice(clust_ix, n_sample)

        for iy, image in enumerate(images[clust_ix]):
            if iy < n_sample:
                axarr[iy, ix].imshow(standardize(image))

    if outfile is not None:
        plt.savefig(outfile)
        return outfile
    else:
        return f, axarr
