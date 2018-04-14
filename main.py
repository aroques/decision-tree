from decision_tree import DecisionTreeClassifier
import argparse


def main():
    args = get_cmd_ln_arguments()

    num_columns, num_rows, training_data = parse_txt_file(args.filename)
    feature_names = get_feature_names(num_columns)

    decision_tree = DecisionTreeClassifier()
    print('Training a decision tree classifier...')
    decision_tree.fit(feature_names, training_data)
    print('Decision tree classifier trained:')
    decision_tree.print()

    while True:
        query = input('Query the decision tree [Y/n]?:').lower()
        if query == '' or query == 'y':
            sample = input('Enter a sample ({} numbers separated by a space): '.format(num_columns))
            sample = line_to_int_list(sample)
            prediction = decision_tree.predict(sample)
            print('Prediction: {}'.format(prediction))
        elif query == 'n':
            break


def get_feature_names(num_features):
    """
    Args:
        num_features: The number of feature names to create

    Returns:
        feature_names: A list of feature names

    """
    feature_names = []
    for i in range(num_features):
        feature_names.append('feature' + str(i + 1))
    return feature_names


def parse_txt_file(file):
    """
    Parses a user supplied text file for data.
    Args:
        file: The name of the text file that contains data.

    Returns:
        num_columns: The number of columns in the training data set
        num_rows: The number of rows in the training data set
        training_data: Training data (a list of integer lists)

    """
    training_data = []
    with open(file) as f:
        first_line = f.readline()
        num_columns = int(first_line[0])
        num_rows = int(first_line[2])
        for line in f:
            data = line_to_int_list(line)
            training_data.append(data)

    return num_columns, num_rows, training_data


def line_to_int_list(line):
    """
    Args:
        line: A string of integers. Ex: '1 3 5\n'

    Returns:
        A list of integers. Ex: [1, 3, 5]
    """
    data = line.split(' ')
    data = [int(x.strip('\n')) for x in data]
    return data


def get_cmd_ln_arguments():
    """
    Returns:
        args: An object that contains command line argument data

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help='name of file that contains data')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
