import matplotlib.pyplot as plt
import pandas as pd

def standardize(i):
    """ Standardize numpy image """
    i = i - i.min()
    i = (i / i.max())
    return i

def plot_clusters(clusters, figsize=(10, 45)):
    cluster_shape = pd.DataFrame(clusters).shape
    f, axarr = plt.subplots(cluster_shape[1], cluster_shape[0], figsize=figsize)
    [axi.set_axis_off() for axi in axarr.ravel()]
    for ix, images in enumerate(clusters):
        for iy, i in enumerate(images):
            axarr[iy, ix].imshow(standardize(images[iy][1]))