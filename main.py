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
    feature_names = training_data.pop(0)
    mytree = DecisionTree(feature_names, training_data)

    mytree.print()

    testing_data = [
        ['Green', 3, 'Apple'],
        ['Yellow', 4, 'Apple'],
        ['Red', 2, 'Grape'],
        ['Red', 1, 'Grape'],
        ['Yellow', 3, 'Lemon'],
    ]

    for row in testing_data:
        print("Actual: %s. Predicted: %s" % (row[-1], mytree.predict(row)))


if __name__ == '__main__':
    main()
