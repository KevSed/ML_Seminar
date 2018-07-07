import pandas as pd
import os
import imageio
import click
from fact.io import to_h5py
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from time import sleep
from skimage.transform import resize
import numpy as np
import h5py


def as_completed(futures):
    futures = list(futures)
    while futures:
        for f in futures.copy():
            if f.ready():
                futures.remove(f)
                yield f.get()
        sleep(0.1)


def read_images(file):
    """
    Load the data and labels from the given folder.
    """
    img_rows = 400
    img_cols = 400
    if file.find('NORMAL') != -1:
        label = 0
    if file.find('CNV') != -1:
        label = 1
    if file.find('DME') != -1:
        label = 2
    if file.find('DRUSEN') != -1:
        label = 3
    data = {}
    img_file = imageio.imread(file)
    img_file = resize(img_file, (img_rows, img_cols))
    data['img_arr'] = np.asarray(img_file, dtype=float)
    data['label'] = label
    return data


@click.command()
@click.argument('input_file', nargs=1, required=True, type=click.Path(exists=False))
@click.argument('output_file', required=True, type=click.Path(exists=False))
@click.option('-n', '--n-jobs', default=cpu_count(), type=int, help='Number of cores to use')
def main(input_file, output_file, n_jobs):

    ordner = os.listdir(input_file)
    files = []
    for ord in ordner:
        for file in os.listdir(input_file + ord):
            if file.endswith('.jpeg'):
                if file is not None:
                    files.append(input_file + ord+ '/' + file)

    gesamt = len(files)

    print("Loading image files using", n_jobs, "cores")
    hdf5_file = h5py.File(output_file, mode='w')
    results = [read_images(f) for f in tqdm(files)]

    hdf5_file.create_dataset("train_img", data=[results[i]['img_arr'] for i in range(gesamt)])
    hdf5_file.create_dataset("train_label", data=[results[i]['label'] for i in range(gesamt)])
    # to_h5py(df, output_file, key='events', mode='a', index=False)
    hdf5_file.close()


if __name__ == '__main__':
    main()
