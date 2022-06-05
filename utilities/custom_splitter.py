import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.utils.validation import check_array


class StratifiedColShuffleSplit(StratifiedShuffleSplit):

    def split(self, X, y, groups=None):
        """
        remark: need argument test_size

        Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

            Note that providing ``y`` is sufficient to generate the splits and
            hence ``np.zeros(n_samples)`` may be used as a placeholder for
            ``X`` instead of actual training data.

        y : Always ignored, exists for compatibility.

        groups : object
            Stratification is done based on the groups.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.

        Notes
        -----
        Randomized CV splitters may return different results for each call of
        split. You can make the results identical by setting `random_state`
        to an integer.
        """

        y = check_array(y, ensure_2d=False, dtype=None)
        return super().split(X, y=groups, groups=None)


class StratifiedColKFold(StratifiedKFold):

    def split(self, X, y, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

            Note that providing ``y`` is sufficient to generate the splits and
            hence ``np.zeros(n_samples)`` may be used as a placeholder for
            ``X`` instead of actual training data.

        y : Always ignored, exists for compatibility.

        groups : object
            Stratification is done based on the groups.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.

        Notes
        -----
        Randomized CV splitters may return different results for each call of
        split. You can make the results identical by setting `random_state`
        to an integer.
        """

        y = check_array(y, ensure_2d=False, dtype=None)
        return super().split(X, y=groups, groups=None)


if __name__ == '__main__':

    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [1, 2], [1, 3], [1, 3], [2, 4]])
    y = np.array([111, 222, 323, 231, -213, -123, -100, -112])
    groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    print(f'for StratifiedColKFold:')
    cv = StratifiedColKFold(n_splits=2)

    for train_index, test_index in cv.split(X, y=y, groups=groups):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print(f'y_train: {y_train}')
        print(f'group: {groups[train_index]}')

    print(f'\n\nfor StratifiedColShuffleSplit:')
    cv = StratifiedColShuffleSplit(n_splits=2,
                                   test_size=0.5
                                   )
    for train_index, test_index in cv.split(X, y=y, groups=groups):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print(f'y_train: {y_train}')
        print(f'group: {groups[train_index]}')

