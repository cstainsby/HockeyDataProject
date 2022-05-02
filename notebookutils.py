from mysklearn import myevaluation
from mysklearn.myclassifiers import MyKNeighborsClassifier, MyNaiveBayesClassifier, MyDummyClassifier, MyDecisionTreeClassifier


def knn_fold(cur_X_train, cur_X_test, X, y, table, keys):
    """ a function to calculate the knn for a fold

    Args:
        cur_X_train - current indecies to train on
        cur_X_test - current indecies to test on
        X - X train
        y - y train
        table - mypytable of data

    Returns:
        Predicted values
    """
    knn = MyKNeighborsClassifier(5)
    knn_X_test = []
    knn_X_train = []

    knn_keys = table.get_key(table.column_names, keys)
    for val in cur_X_train:
        instance = []
        for key in knn_keys:
            instance.append(X[val][key])
        knn_X_train.append(instance)
    for val in cur_X_test:
        instance = []
        for key in knn_keys:
            instance.append(X[val][key])
        knn_X_test.append(instance)

    cur_y_train = []
    for val in cur_X_train:
        cur_y_train.append(y[val])



    knn.fit(knn_X_train, cur_y_train)
    response = knn.predict(knn_X_test)
    return response


def descision_tree(cur_X_train, cur_X_test, table, X, y, keys):
    dt = MyDecisionTreeClassifier()
    X_test = []
    X_train = []
    tree_keys = table.get_key(table.column_names, keys)
    for val in cur_X_train:
        instance = []
        for key in tree_keys:
            instance.append(X[val][key])
        X_train.append(instance)
    for val in cur_X_test:
        instance = []
        for key in tree_keys:
            instance.append(X[val][key])
        X_test.append(instance)

    cur_y_train = []
    for val in cur_X_train:
        cur_y_train.append(y[val])

    dt.fit(X_train, cur_y_train)
    response = dt.predict(X_test)
    return response


def kfoldstrat(table, X, y, identifier, keys):
    if identifier == "knn":
        train_folds, test_folds = myevaluation.stratified_kfold_cross_validation(X, y, 10)
        complete = []
        true = []
        for fold in test_folds:
            for val in fold:
                true.append(y[val])
        for x in range(len(train_folds)):
            pred = knn_fold(train_folds[x], test_folds[x], X, y, table,keys)
            for val in pred:
                complete.append(val)
        strat_knn_pred = complete
        strat_knn_true = true
        return true, complete
    elif identifier == "naive":
        train_folds, test_folds = myevaluation.stratified_kfold_cross_validation(X, y, 10)
        complete = []
        true = []
        for fold in test_folds:
            for val in fold:
                true.append(y[val])
        for x in range(len(train_folds)):
            pred = naivebayes(train_folds[x], test_folds[x], table, X, y, keys)
            for val in pred:
                complete.append(val)
        strat_dummy_pred = complete
        strat_dummy_true = true
        return true, complete
    else:
        train_folds, test_folds = myevaluation.stratified_kfold_cross_validation(X, y, 10)
        complete = []
        true = []
        for fold in test_folds:
            for val in fold:
                true.append(y[val])
        for x in range(len(train_folds)):
            pred = descision_tree(train_folds[x], test_folds[x], table, X, y, keys)
            for val in pred:
                complete.append(val)
        strat_dummy_pred = complete
        strat_dummy_true = true
        return true, complete


def naivebayes(cur_X_train, cur_X_test, table, X, y, keys):
    nb = MyNaiveBayesClassifier()
    X_test = []
    X_train = []
    naive_keys = table.get_key(table.column_names, keys)
    for val in cur_X_train:
        instance = []
        for key in naive_keys:
            instance.append(X[val][key])
        X_train.append(instance)
    for val in cur_X_test:
        instance = []
        for key in naive_keys:
            instance.append(X[val][key])
        X_test.append(instance)

    cur_y_train = []
    for val in cur_X_train:
        cur_y_train.append(y[val])


    nb.fit(X_train, cur_y_train)
    response = nb.predict(X_test)
    return response

def matrix(true, pred):
    str_true = []
    str_pred = []
    for x in range(len(pred)):
        str_true.append(str(true[x]))
        str_pred.append(str(pred[x]))
    conf_matrix = myevaluation.confusion_matrix(str_true, str_pred, ["H","A"])
    for x in range(len(conf_matrix)):
        total = sum(conf_matrix[x])
        correct = conf_matrix[x][x]
        if x == 0:
            conf_matrix[x].insert(0,"Won")
        else:
            conf_matrix[x].insert(0,"Lost")
        conf_matrix[x].append(total)
        if correct == 0:
            conf_matrix[x].append(0)
        else:
            conf_matrix[x].append(round(correct/total,3)*100)

    return conf_matrix


def discretize_ptspercent(table, index):
    for row in table:
        val = float(row[index])
        if val < .1:
            row[index] = '1'
        elif val < .2:
            row[index] = '2'
        elif val < .3:
            row[index] = '3'
        elif val < .4:
            row[index] = '4'
        elif val < .5:
            row[index] = '5'
        elif val < .6:
            row[index] = '6'
        elif val < .7:
            row[index] = '7'
        elif val < .8:
            row[index] = '8'
        elif val < .9:
            row[index] = '9'
        elif val < 1:
            row[index] = '10'
