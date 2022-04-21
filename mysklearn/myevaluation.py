import math
import numpy as np

from mysklearn import myutils

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
            Use random_state to seed your random number generator
                you can use the math module or use numpy for your generator
                choose one and consistently use that generator throughout your code
        shuffle(bool): whether or not to randomize the order of the instances before splitting
            Shuffle the rows in X and y before splitting and be sure to maintain the parallel order of X and y!!

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)

    Note:
        Loosely based on sklearn's train_test_split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    index_list = []

    if shuffle:
        index_list = myutils.randomized_index_list(len(y), random_state)
    else:
        index_list = [i for i in range(len(y))]
    
    if type(test_size) is int:
        test_size = float(test_size)/len(y)

    for i, index in enumerate(index_list):
        print(len(y) - math.ceil(len(y) * test_size))
        if i < len(y) - math.ceil(len(y) * test_size):
            # while in the range of train data add to train list
            X_train.append(X[index])
            y_train.append(y[index])
        else:
            # while in test range, add to test list
            X_test.append(X[index])
            y_test.append(y[index])
            
    return X_train, X_test, y_train, y_test

def kfold_cross_validation(X, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold
        X_test_folds(list of list of int): The list of testing set indices for each fold

    Notes:
        The first n_samples % n_splits folds have size n_samples // n_splits + 1,
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    index_list = []
    if shuffle:
        index_list = myutils.randomized_index_list(len(X), random_state)
    else:
        index_list = [i for i in range(len(X))]

    X_train_folds = [[] for i in range(n_splits)]
    X_test_folds = [[] for i in range(n_splits)]

    # find the size of each fold
    fold_sizes = []
    for i in range(n_splits):
        if i < len(X) % n_splits:
            fold_sizes.append(len(X) // n_splits + 1)
        else:
            fold_sizes.append(len(X) // n_splits)

    # split the indices into subsets based off fold sizes
    subsets = [[] for i in range(n_splits)]
    index_in_index_list = 0
    for i, size in enumerate(fold_sizes):
        for j in range(index_in_index_list,  size + index_in_index_list):
            subsets[i].append(index_list[j])
        index_in_index_list += size

    for k in range(n_splits):
        X_test_folds[k] = subsets[k]
        X_train_folds[k] = myutils.combine_lists(subsets, [k])

    return X_train_folds, X_test_folds 

def stratified_kfold_cross_validation(X, y, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples).
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X).
            The shape of y is n_samples
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold.
        X_test_folds(list of list of int): The list of testing set indices for each fold.

    Notes:
        Loosely based on sklearn's StratifiedKFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    groups = []
    seen = []

    if random_state is not None:
        myutils.seed(random_state)

    if shuffle:
        indecies = list(range(len(X)))
        myutils.randomize_in_place(indecies)
    else:
        indecies = list(range(len(X)))

    for index in indecies:
        if y[index] in seen:
            seen_index = seen.index(y[index])
            groups[seen_index].append(index)
        else:
            seen.append(y[index])
            groups.append([index])


    deck = []

    for row in groups:
        for val in row:
            deck.append(val)

    train_folds = []
    test_folds = []

    for x in range(n_splits):
        train_folds.append([])
        test_folds.append([])

    for x in range(len(deck)):
        test_folds[x % n_splits].append(deck[x])

    for x in range(len(test_folds)):
        for val in deck:
            if val not in test_folds[x]:
                train_folds[x].append(val)

    return train_folds, test_folds

def bootstrap_sample(X, y=None, n_samples=None, random_state=None):
    """Split dataset into bootstrapped training set and out of bag test set.

    Args:
        X(list of list of obj): The list of samples
        y(list of obj): The target y values (parallel to X)
            Default is None (in this case, the calling code only wants to sample X)
        n_samples(int): Number of samples to generate. If left to None (default) this is automatically
            set to the first dimension of X.
        random_state(int): integer used for seeding a random number generator for reproducible results
    Returns:
        X_sample(list of list of obj): The list of samples
        X_out_of_bag(list of list of obj): The list of "out of bag" samples (e.g. left-over samples)
        y_sample(list of obj): The list of target y values sampled (parallel to X_sample)
            None if y is None
        y_out_of_bag(list of obj): The list of target y values "out of bag" (parallel to X_out_of_bag)
            None if y is None
    Notes:
        Loosely based on sklearn's resample():
            https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
    """
    np.random.seed(random_state)
    
    X_sample = []
    X_out_of_bag = []
    y_sample = []
    y_out_of_bag = []
    indices_in_bag = []

    if n_samples is not None:
        index_list = [np.random.randint(0, len(X)) for i in range(n_samples)]
    else:
        index_list = [np.random.randint(0, len(X)) for i in range(len(y))]

    for i, index in enumerate(index_list):
        indices_in_bag.append(index)
        X_sample.append(X[index])
        if y is not None:
            y_sample.append(y[index])
    
    for i in range(len(X)):
        if indices_in_bag.count(i) == 0:
            X_out_of_bag.append(X[i])
            if y is not None:
                y_out_of_bag.append(y[i])
    
    if y is None:
        y_sample = None
        y_out_of_bag = None

    return X_sample, X_out_of_bag, y_sample, y_out_of_bag
            

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    confusion_matrix = [([0] * len(labels)) for i in range(len(labels))]

    for i, predicted_value in enumerate(y_pred):
        predicted_index = labels.index(predicted_value)
        actual_index = labels.index(y_true[i])

        confusion_matrix[actual_index][predicted_index] += 1

    return confusion_matrix 

def accuracy_score(y_true, y_pred, normalize=True):
    """Compute the classification prediction accuracy score.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        normalize(bool): If False, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.

    Returns:
        score(float): If normalize == True, return the fraction of correctly classified samples (float),
            else returns the number of correctly classified samples (int).

    Notes:
        Loosely based on sklearn's accuracy_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    """
    # (TP + TN)/(TP + TN + FP + FN)
    accuracy = 0.0

    for i, predicted_value in enumerate(y_pred):
        if predicted_value == y_true[i]:
            accuracy += 1.0

    if normalize:
        accuracy /= len(y_pred)

    return accuracy

def binary_precision_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the precision (for binary classification). The precision is the ratio tp / (tp + fp)
        where tp is the number of true positives and fp the number of false positives.
        The precision is intuitively the ability of the classifier not to label as
        positive a sample that is negative. The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        precision(float): Precision of the positive class

    Notes:
        Loosely based on sklearn's precision_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    """
    if labels is None:
        item_label_list, parallel_frequency_list = myutils.find_frequency_of_each_element_in_list(y_true)
        labels = item_label_list

    if pos_label is None:
        pos_label = labels[0]

    tp_count = 0
    fp_count = 0

    for i, prediction in enumerate(y_pred):
        if prediction == y_true[i] and prediction == pos_label:
            tp_count += 1
        elif prediction != y_true[i] and prediction == pos_label:
            fp_count += 1

    numer = tp_count
    denom = (tp_count + fp_count)
    if denom == 0:
        return 0.0
    return numer/denom

def binary_recall_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the recall (for binary classification). The recall is the ratio tp / (tp + fn) where tp is
        the number of true positives and fn the number of false negatives.
        The recall is intuitively the ability of the classifier to find all the positive samples.
        The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        recall(float): Recall of the positive class

    Notes:
        Loosely based on sklearn's recall_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    """
    if labels is None:
        item_label_list, parallel_frequency_list = myutils.find_frequency_of_each_element_in_list(y_true)
        labels = item_label_list

    if pos_label is None:
        pos_label = labels[0]

    tp_count = 0
    fn_count = 0

    for i, prediction in enumerate(y_pred):
        if prediction == y_true[i] and prediction == pos_label:
            tp_count += 1
        elif prediction != y_true[i] and prediction != pos_label:
            fn_count += 1

    numer = tp_count
    denom = (tp_count + fn_count)
    if denom == 0:
        return 0.0
    return numer/denom

def binary_f1_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the F1 score (for binary classification), also known as balanced F-score or F-measure.
        The F1 score can be interpreted as a harmonic mean of the precision and recall,
        where an F1 score reaches its best value at 1 and worst score at 0.
        The relative contribution of precision and recall to the F1 score are equal.
        The formula for the F1 score is: F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        f1(float): F1 score of the positive class

    Notes:
        Loosely based on sklearn's f1_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """
    precision = binary_precision_score(y_true, y_pred, labels, pos_label)
    recall = binary_recall_score(y_true, y_pred, labels, pos_label)
    if precision == 0 and recall == 0:
        return 0.0
    return 2*(precision * recall) / (precision + recall)
