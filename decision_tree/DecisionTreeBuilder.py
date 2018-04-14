from .Question import Question
from .DecisionNode import DecisionNode
from .Leaf import Leaf
from .dataset_utils import class_counts, unique_vals


class DecisionTreeBuilder:

    def __init__(self, headers, rows):
        self.headers = headers
        self.rows = rows

    def build(self):
        tree = self.build_tree(self.rows)
        return tree

    def build_tree(self, rows):
        """Builds the tree.

        Rules of recursion: 1) Believe that it works. 2) Start by checking
        for the base case (no further information gain). 3) Prepare for
        giant stack traces.
        """

        # Try partitioning the dataset on each of the unique attribute,
        # calculate the information gain,
        # and return the question that produces the highest gain.
        gain, question = self.__find_best_split(rows)

        # Base case: no further info gain
        # Since we can ask no further questions,
        # we'll return a leaf.
        if gain == 0:
            return Leaf(rows)

        # If we reach here, we have found a useful feature / value
        # to partition on.
        true_rows, false_rows = DecisionTreeBuilder.__partition(rows, question)

        # Recursively build the true branch.
        true_branch = self.build_tree(true_rows)

        # Recursively build the false branch.
        false_branch = self.build_tree(false_rows)

        # Return a Question node.
        # This records the best feature / value to ask at this point,
        # as well as the branches to follow
        # depending on the answer.
        return DecisionNode(question, true_branch, false_branch)

    def __find_best_split(self, rows):
        """Find the best question to ask by iterating over every feature / value
        and calculating the information gain."""
        best_gain = 0  # keep track of the best information gain
        best_question = None  # keep train of the feature / value that produced it
        current_uncertainty = self.__gini(rows)
        n_features = len(rows[0]) - 1  # number of columns

        for col in range(n_features):  # for each feature

            values = unique_vals(rows, col)  # unique values in the column

            for val in values:  # for each value

                question = Question(self.headers, col, val)

                # try splitting the dataset
                true_rows, false_rows = self.__partition(rows, question)

                # Skip this split if it doesn't divide the
                # dataset.
                if len(true_rows) == 0 or len(false_rows) == 0:
                    continue

                # Calculate the information gain from this split
                gain = self.__info_gain(true_rows, false_rows, current_uncertainty)

                # You actually can use '>' instead of '>=' here
                # but I wanted the tree to look a certain way for our
                # toy dataset.
                if gain >= best_gain:
                    best_gain, best_question = gain, question

        return best_gain, best_question

    def __info_gain(self, left, right, current_uncertainty):
        """Information Gain.

        The uncertainty of the starting node, minus the weighted impurity of
        two child nodes.
        """
        total_len = len(left) + len(right)
        left_weight = float(len(left)) / total_len
        right_weight = 1 - left_weight
        return current_uncertainty - left_weight * self.__gini(left) - right_weight * self.__gini(right)

    @staticmethod
    def __gini(rows):
        """Calculate the Gini Impurity for a list of rows.

        There are a few different ways to do this, I thought this one was
        the most concise. See:
        https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
        """
        counts = class_counts(rows)
        impurity = 1
        for lbl in counts:
            prob_of_lbl = counts[lbl] / float(len(rows))
            impurity -= prob_of_lbl ** 2
        return impurity

    @staticmethod
    def __partition(rows, question):
        """Partitions a dataset.

        For each row in the dataset, check if it matches the question. If
        so, add it to 'true rows', otherwise, add it to 'false rows'.
        """
        true_rows, false_rows = [], []
        for row in rows:
            if question.match(row):
                true_rows.append(row)
            else:
                false_rows.append(row)
        return true_rows, false_rows
