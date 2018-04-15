from .dataset_utils import class_counts


class Leaf:
    """
    A Leaf node classifies data.

    This holds a dictionary:
        Key: class (e.g., "Apple")
        Value: Number of times the class appears in the rows
    """

    def __init__(self, rows):
        """
        Initializes a Leaf Node

        self.predictions is a dictionary to handle the cases where there is a equal chance that the
        class could be one of many classes. For example, imagine you have training data with exactly
        two examples that contain the same feature values, but different class labels like:
            - ['Yellow', 3, 'Lemon']
            - ['Yellow', 3, 'Apple']
        Then, there may be a leaf node so that yellow fruit with diameter of 3
        could either be classified as a Lemon or Apple. Both are equally likely.

        Args:
            rows: Rows used to create the Leaf

        """
        self.predictions = class_counts(rows)

    def __repr__(self):
        """
        Returns:
            A readable representation of a leaf
        """
        probs = {}
        for lbl, val in self.probabilities.items():
            probs[lbl] = str(int(val * 100)) + "%"
        return str(probs)

    @property
    def probabilities(self):
        """
        Returns:
            Dictionary: Key: class label. Value: probability (data type: float)

        """
        total = sum(self.predictions.values()) * 1.0
        probs = {}
        for lbl in self.predictions.keys():
            probs[lbl] = self.predictions[lbl] / total
        return probs

    def prediction(self):
        """
        Returns:
            Returns prediction with dictionary that contains the most
            certain prediction.

        """
        prediction = {}
        highest_prob = 0
        key = ''
        for k, v in self.probabilities.items():
            if v > highest_prob:
                key = k
                highest_prob = v
        prediction[key] = str(int(highest_prob * 100)) + "%"
        return prediction
