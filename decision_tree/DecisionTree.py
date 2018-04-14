from .DecisionTreeBuilder import DecisionTreeBuilder
from .Leaf import Leaf


class DecisionTreeClassifier:
    """
    A CART Decision Tree Classifier
    """
    def __init__(self):
        self.tree = None

    def fit(self, feature_names, rows):
        tree_builder = DecisionTreeBuilder()
        self.tree = tree_builder.build(feature_names, rows)

    def predict(self, row):
        """
        Calls recursive classify function to produce a class prediction.
        Args:
            row: A row to be predicted

        Returns:
            A Leaf Node (class prediction)

        """
        prediction = self.classify(row, self.tree)
        return prediction

    def classify(self, row, node):
        """
        Recursively classifies a row
        Args:
            row: The row to classify
            node: Developer passes in the root node of the tree,
                but the function will be recursively called with descending Decision Nodes
                until a Leaf Node is returned.

        Returns:
            A Leaf Node

        """
        # Base case: We've reached a leaf
        if isinstance(node, Leaf):
            return node

        # Node is an instance of a Decision Node

        # Use the question stored in the Decision Node to
        # traverse down either the true of false branch of the Decision Tree
        if node.question.match(row):
            return self.classify(row, node.true_branch)
        else:
            return self.classify(row, node.false_branch)

    def print(self):
        """
        Prints the Decision tree.
        Returns:
            None

        """
        self.print_tree(self.tree)

    def print_tree(self, node, spacing=""):
        """
        Uses recursion to print the decision tree.
        Args:
            node: Starting node (pass the root to print the whole tree)
            spacing: Left-padding (indentation)

        Returns:
            None

        """
        # Base case: we've reached a leaf
        if isinstance(node, Leaf):
            print(spacing + "Predict", node)
            return

        # Node is a Decision Node so print the question at this node
        print(spacing + str(node.question))

        # Call this function recursively on the true branch
        print(spacing + '--> True:')
        self.print_tree(node.true_branch, spacing + "  ")

        # Call this function recursively on the false branch
        print(spacing + '--> False:')
        self.print_tree(node.false_branch, spacing + "  ")
