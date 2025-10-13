from keras.datasets import cifar10
import numpy as np

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
y_train = y_train.flatten()
y_test = y_test.flatten()

def select_balanced_subset(X, y, samples_per_class=500):
    selected_X = []
    selected_y = []
    for class_label in np.unique(y):
        idx = np.where(y == class_label)[0]
        selected_idx = np.random.choice(idx, samples_per_class, replace=False)
        selected_X.append(X[selected_idx])
        selected_y.append(y[selected_idx])
    return np.concatenate(selected_X), np.concatenate(selected_y)

X_train_small, y_train_small = select_balanced_subset(X_train, y_train, samples_per_class=500)
X_test_small, y_test_small = select_balanced_subset(X_test, y_test, samples_per_class=100)
