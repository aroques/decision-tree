import os
import csv
from config import ROOT_DIR


def load_fruit():
    return _load_dataset('fruit.csv')


def _load_dataset(filename):
    return load_csv(_get_path(filename))


def _get_path(filename):
    path = os.path.join(ROOT_DIR, 'datasets', 'data', filename)
    return os.path.abspath(path)


def load_csv(filename):
    """
    Load CSV data from a file and convert the attributes that can be converted to numbers.
    :param filename: The name of the CSV file to load.
    :return: A list of the dataset.
    """
    f = open(filename)
    lines = csv.reader(f)
    dataset = list(lines)
    for i in range(len(dataset)):
        for j in range(len(dataset[0])):
            try:
                dataset[i][j] = dataset[i][j].strip()
                dataset[i][j] = float(dataset[i][j])
            except ValueError:
                pass

    f.close()
    return dataset
