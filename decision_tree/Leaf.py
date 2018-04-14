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

    def __repr__(self):
        """
        Returns:
            A readable representation of a leaf
        """
        total = sum(self.predictions.values()) * 1.0
        probs = {}
        for lbl in self.predictions.keys():
            probs[lbl] = str(int(self.predictions[lbl] / total * 100)) + "%"
        return str(probs)
