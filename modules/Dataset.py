import csv
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class Dataset:
    X_train = None
    X_test = None
    y_train = None
    y_test = None

    @staticmethod
    def one_hot_encode(arr):
        unique_elements = np.unique(arr)
        element_to_one_hot = {elem: np.eye(len(unique_elements))[i] for i, elem in enumerate(unique_elements)}
        return np.array([element_to_one_hot[elem] for elem in arr])

    @staticmethod
    def label_encode(arr):
        le = LabelEncoder()
        return np.apply_along_axis(le.fit_transform, 0, arr)

    @staticmethod
    def shuffle_data(X, y):
        assert len(X) == len(y)
        p = np.random.permutation(len(X))
        return X[p], y[p]

    @staticmethod
    def set_iris():
        X, y = Dataset.load_csv('datasets/iris/iris.data', float)
        y = Dataset.one_hot_encode(y)
        Dataset.shuffle_data(X,y)
        Dataset.X_train, Dataset.X_test, Dataset.y_train, Dataset.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    @staticmethod
    def set_tictactoe():
        X, y = Dataset.load_csv('datasets/tic+tac+toe+endgame/tic-tac-toe.data', str)
        X = Dataset.label_encode(X)
        y = Dataset.one_hot_encode(y)
        Dataset.shuffle_data(X,y)
        Dataset.X_train, Dataset.X_test, Dataset.y_train, Dataset.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    @staticmethod
    def load_csv(path, dtype):
        X = []
        y = []
        with open(path, 'r') as file:
            csv_reader = csv.reader(file)
            # No header skipping here
            for row in csv_reader:
                if len(row) == 0:
                    continue  # Skip empty rows
                if len(row) < 2:
                    raise ValueError(f"Row doesn't have enough columns: {row}")

                features = list(map(dtype, row[:-1]))  # Assuming features are in all columns except the last one
                label = row[-1]  # Assuming the label is in the last column
                X.append(features)
                y.append(label)

        return np.array(X), np.array(y)

