from .Question import Question
from .DecisionNode import DecisionNode
from .Leaf import Leaf
from .dataset_utils import class_counts, unique_vals


class DecisionTreeBuilder:
    """
    An object that builds CART Decision Trees
    """

    def __init__(self):
        self.feature_names = None
        self.rows = None

    def build(self, feature_names, rows):
        """
        Builds a decision tree.
        Args:
            feature_names: The names of the features (column headers)
            rows: Training data (X and Y)

        Returns:
            A Decision Tree

        """
        self.feature_names = feature_names
        self.rows = rows
        tree = self.build_tree(self.rows)
        return tree

    def build_tree(self, rows):
        """
        Recursively builds a decision tree.
        Args:
            rows: Rows in a dataset

        Returns:
            A decision tree

        """

        gain, question = self.__find_best_split(rows)

        if gain == 0:
            # Base case: No more information gain, so return a leaf
            return Leaf(rows)

        # Use question to partition rows
        true_rows, false_rows = self.__partition(rows, question)

        # Recursively build the true and false branches
        true_branch = self.build_tree(true_rows)
        false_branch = self.build_tree(false_rows)

        # The information gain is greater than 0 so return a decision node
        return DecisionNode(question, true_branch, false_branch)

    def __find_best_split(self, rows):
        """
        Find the best question to ask by iterating over every feature / value
        and calculating the information gain.
        Args:
            rows: Rows of a dataset

        Returns:
            best_gain: The amount of information gain that best_question produces
            best_question: The question that produces the most information gain

        """
        best_gain = 0
        best_question = None
        current_uncertainty = self.__gini(rows)
        num_features = len(rows[0]) - 1

        for col in range(num_features):

            # Get list of the unique values in the current column
            values = unique_vals(rows, col)

            for val in values:

                question = Question(self.feature_names, col, val)

                true_rows, false_rows = self.__partition(rows, question)

                if len(true_rows) == 0 or len(false_rows) == 0:
                    # The split does not divide the dataset so skip it
                    continue

                gain = self.__info_gain(true_rows, false_rows, current_uncertainty)

                if gain >= best_gain:
                    best_gain, best_question = gain, question

        return best_gain, best_question

    def __info_gain(self, left, right, current_uncertainty):
        """
        Information Gain.

        The uncertainty of the starting node, minus the weighted impurity of
        two child nodes.
        Args:
            left: left partition of a dataset
            right: right partition of a dataset
            current_uncertainty: The uncertainty before splitting

        Returns:
            Information gained from splitting

        """
        total_len = len(left) + len(right)
        left_weight = float(len(left)) / total_len
        right_weight = 1 - left_weight
        return current_uncertainty - left_weight * self.__gini(left) - right_weight * self.__gini(right)

    @staticmethod
    def __gini(rows):
        """
        Calculate the Gini Impurity for a list of rows.
        For more information see:
          https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
        Args:
            rows: Rows of a dataset

        Returns:
            The Gini Impurity of the list of rows

        """
        counts = class_counts(rows)
        impurity = 1
        for lbl in counts:
            prob_of_lbl = counts[lbl] / float(len(rows))
            impurity -= prob_of_lbl ** 2
        return impurity

    @staticmethod
    def __partition(rows, question):
        """
        Partitions a dataset.

        Args:
            rows: Rows of a dataset
            question: The question to split the rows on

        Returns:
            true_rows: Rows that match the question
            false_rows: Rows that do not match the question

        """
        true_rows, false_rows = [], []
        for row in rows:
            if question.match(row):
                true_rows.append(row)
            else:
                false_rows.append(row)
        return true_rows, false_rows
