"""A module for running a KNN on Histogram of Oriented Gradients input."""
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

def knn(X_train, y_train, X_test, y_test):
    n_samples, nx, ny = X_train.shape
    X_train = X_train.reshape((n_samples, nx * ny))
    y_train = y_train.reshape((y_train.shape[0]))
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train, y_train)

    n_samples, nx, ny = X_test.shape
    X_test = X_test.reshape((n_samples, nx * ny))
    y_test = y_test.reshape((y_test.shape[0]))

    y_pred = neigh.predict(X_test)

    print(accuracy_score(y_test, y_pred))


if __name__ == '__main__':
    X_train = np.load('data/hog/train/hog_123.npy')
    y_train = np.load('data/hog/train/labels.npy')

    X_test = np.load('data/hog/test/hog_123.npy')
    y_test = np.load('data/hog/test/labels.npy')

    knn(X_train, y_train, X_test, y_test)

    X_train = np.load('data/hog/train/hog_678.npy')
    y_train = np.load('data/hog/train/labels.npy')

    X_test = np.load('data/hog/test/hog_678.npy')
    y_test = np.load('data/hog/test/labels.npy')

    knn(X_train, y_train, X_test, y_test)

    X_train = np.load('data/hog/train/hog_111213.npy')
    y_train = np.load('data/hog/train/labels.npy')

    X_test = np.load('data/hog/test/hog_111213.npy')
    y_test = np.load('data/hog/test/labels.npy')

    knn(X_train, y_train, X_test, y_test)


