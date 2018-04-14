from .DecisionTreeBuilder import DecisionTreeBuilder
from .Leaf import Leaf


class DecisionTree:
    def __init__(self, headers, rows):
        tree_builder = DecisionTreeBuilder(headers, rows)
        self.tree = tree_builder.build()

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
