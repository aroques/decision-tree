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
        prediction = self.classify(row, self.tree)
        return prediction

    def classify(self, row, node):
        """See the 'rules of recursion' above."""

        # Base case: we've reached a leaf
        if isinstance(node, Leaf):
            return node

        # Decide whether to follow the true-branch or the false-branch.
        # Compare the feature / value stored in the node,
        # to the example we're considering.
        if node.question.match(row):
            return self.classify(row, node.true_branch)
        else:
            return self.classify(row, node.false_branch)

    def print(self):
        self.print_tree(self.tree)

    def print_tree(self, node, spacing=""):
        """World's most elegant tree printing function."""

        # Base case: we've reached a leaf
        if isinstance(node, Leaf):
            print(spacing + "Predict", node.predictions)
            return

        # Print the question at this node
        print(spacing + str(node.question))

        # Call this function recursively on the true branch
        print(spacing + '--> True:')
        self.print_tree(node.true_branch, spacing + "  ")

        # Call this function recursively on the false branch
        print(spacing + '--> False:')
        self.print_tree(node.false_branch, spacing + "  ")
