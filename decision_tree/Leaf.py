from .dataset_utils import class_counts


class Leaf:
    """
    A Leaf node classifies data.

    This holds a dictionary:
        Key: class (e.g., "Apple")
        Value: Number of times the class appears in the rows
    """

    def __init__(self, rows):
        self.predictions = class_counts(rows)
