from decision_tree import DecisionTreeClassifier


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
    mytree = DecisionTreeClassifier()
    mytree.fit(feature_names, training_data)

    mytree.print()

    testing_data = [
        ['Green', 3, 'Apple'],
        ['Yellow', 4, 'Apple'],
        ['Red', 2, 'Grape'],
        ['Red', 1, 'Grape'],
        ['Yellow', 3, 'Lemon'],
    ]

    actual_values = []
    for i, row in enumerate(testing_data):
        actual_values.append(row.pop())

    for i, row in enumerate(testing_data):
        print("Actual: %s. Predicted: %s" % (actual_values[i], mytree.predict(row)))


if __name__ == '__main__':
    main()
