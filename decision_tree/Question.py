from .utils import is_numeric


class Question:
    """
    A Question is used to partition a dataset.

    This class just records a 'column number' (e.g., 0 for Color) and a
    'column value' (e.g., Green). The 'match' method is used to compare
    the feature value in an example to the feature value stored in the
    question.
    """

    def __init__(self, labels, column, value):
        """
        Initializes a Question
        Args:
            labels: The labels (column headers) of a dataset
            column: The column index of value
            value: Value that will be compared to examples
        """
        self.label = labels[column]
        self.column = column
        self.value = value

    def match(self, example):
        """
        Compares the value of the Question with an example
        Args:
            example: The example that will be compared

        Returns:
            If numeric it returns whether or not the example is greater than or equal to value
            Else it returns whether or not the example is equivalent to value

        """
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        """
        Returns:
            A readable representation of the Question
        """
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            self.label, condition, str(self.value))
