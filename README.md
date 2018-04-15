# Decision Tree

The Decision Tree Builder is responsible for building the decision tree. 

The tree is built with a recursive algorithm. If a Leaf node contains classes that have an equal probability than the first class label in the Leaf's prediction dictionary is arbitrarily chosen.

The tree is general enough to work on numeric data that is not binary. If given binary data the tree will still properly predict class labels, however the 'questions' may appear strange since they may be phrased 'Is feature1 >= 1'.

The program will be expecting a text file that contains training data. Below is an example of such a text file. 3 is the number of features and 5 is the number of rows in the dataset. The rest of the rows are the actual training data and the last column in each of the rows is the row's class label.

```
3 5
1 1 1 0
1 1 0 1
0 0 1 1
1 1 0 0
1 0 0 1
```

#### To run this program:
```
python3.6 main.py filename
```

#### For help run:
```
python3.6 main.py -h
```
