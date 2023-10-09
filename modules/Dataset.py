import csv
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

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
    def set_thyroiddisease():
        # Load the dataset
        X, y = Dataset.load_csv('datasets/thyroid+disease/ann-train.data', float)

        # One-hot encode the labels using your custom method
        y = Dataset.one_hot_encode(y)

        # Shuffle the data
        X, y = Dataset.shuffle_data(X, y)

        # Split the dataset into training and testing sets
        Dataset.X_train, Dataset.X_test, Dataset.y_train, Dataset.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    @staticmethod
    def set_statlogshuttle():
        # Load the dataset
        X, y = Dataset.load_csv('datasets/statlog+shuttle/shuttle.trn', int)

        # One-hot encode the labels using your custom method
        y = Dataset.one_hot_encode(y)

        # Shuffle the data
        X, y = Dataset.shuffle_data(X, y)

        # Split the dataset into training and testing sets
        Dataset.X_train, Dataset.X_test, Dataset.y_train, Dataset.y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    @staticmethod
    def set_oceansync():
        X, y = Dataset.load_csv('datasets/oceansync/GWY1839.csv', float)
        
        # Shuffle the data
        Dataset.shuffle_data(X, y)
        
        # Split the data into training and test sets
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

                try:
                    features = list(map(dtype, row[:-1]))  # Assuming features are in all columns except the last one
                except ValueError:
                    # Skip the first row if it contains non-convertible strings (likely a header)
                    continue

                label = row[-1]  # Assuming the label is in the last column
                X.append(features)
                y.append(label)

        return np.array(X), np.array(y)


    @staticmethod
    def resample_data(heuristic=1, tau=200):
        if tau > Dataset.y_train.shape[0]:
            print(f"Sample size tau={tau} is larger than the number of available instances. Using all instances instead.")
            return

        unique_classes = np.unique(np.argmax(Dataset.y_train, axis=1))
        if heuristic == 1:
            # Uniform sampling without replacement
            indices = np.random.choice(Dataset.y_train.shape[0], tau, replace=False)
        else:
            # Class-wise sampling
            indices = []
            per_class_count = tau // len(unique_classes)
            for cls in unique_classes:
                cls_indices = np.where(np.argmax(Dataset.y_train, axis=1) == cls)[0]
                sampled_indices = resample(cls_indices, n_samples=per_class_count, replace=len(cls_indices) < per_class_count)
                indices.extend(sampled_indices)
        Dataset.X_train = Dataset.X_train[indices]
        Dataset.y_train = Dataset.y_train[indices]

