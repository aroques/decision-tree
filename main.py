from __future__ import print_function
from decision_tree import DecisionTree


def main():
    training_data = [
        ["color", "diameter", "label"],
        ['Green', 3, 'Apple'],
        ['Yellow', 3, 'Apple'],
        ['Red', 1, 'Grape'],
        ['Red', 1, 'Grape'],
        ['Yellow', 3, 'Lemon'],
    ]
    headers = training_data.pop(0)
    mytree = DecisionTree(headers, training_data)

    mytree.print()


if __name__ == '__main__':
    main()
