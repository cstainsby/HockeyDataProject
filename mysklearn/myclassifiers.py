import math
import operator
import numpy as np
import os

from mysklearn import myevaluation, myutils
from mysklearn.mysimplelinearregressor import MySimpleLinearRegressor
import mysklearn

class MySimpleLinearRegressionClassifier:
    """Represents a simple linear regression classifier that discretizes
        predictions from a simple linear regressor (see MySimpleLinearRegressor).

    Attributes:
        discretizer(function): a function that discretizes a numeric value into
            a string label. The function's signature is func(obj) -> obj
        regressor(MySimpleLinearRegressor): the underlying regression model that
            fits a line to x and y data

    Notes:
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self, discretizer, regressor=None):
        """Initializer for MySimpleLinearClassifier.

        Args:
            discretizer(function): a function that discretizes a numeric value into
                a string label. The function's signature is func(obj) -> obj
            regressor(MySimpleLinearRegressor): the underlying regression model that
                fits a line to x and y data (None if to be created in fit())
        """
        self.discretizer = discretizer
        self.regressor = regressor

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        lin_reg = MySimpleLinearRegressor()
        lin_reg.fit(X_train, y_train)

        self.regressor = lin_reg

    def predict(self, X_test):
        """Makes predictions for test samples in X_test by applying discretizer
            to the numeric predictions from regressor.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []

        predictions = self.regressor.predict(X_test)

        for element in predictions:
            discretized_prediction = self.discretizer(element)
            y_predicted.append(discretized_prediction)

        return y_predicted

class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train

    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        distances = []
        neighbor_indices = []

        for test_i, test_instance in enumerate(X_test):
            distances.append([])
            neighbor_indices.append([])

            distance_and_index_for_each_train = []
            # find distances for each index 
            for row, train_instance in enumerate(self.X_train):
                d = myutils.compute_euclidean_distance(train_instance, test_instance)
                distance_and_index_for_each_train.append([d, row])

            # find the closest instances
            distance_and_index_for_each_train.sort(key=operator.itemgetter(0))
            
            for i in range(self.n_neighbors):
                distances[test_i].append(distance_and_index_for_each_train[i][0])
                neighbor_indices[test_i].append(distance_and_index_for_each_train[i][1])

        return distances, neighbor_indices

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []

        distances, neighbor_indices = self.kneighbors(X_test)
        
        for test_i in range(len(X_test)):
            votes = []
            for i in range(len(neighbor_indices[test_i])):
                # add the neighbors classification
                votes.append(self.y_train[neighbor_indices[test_i][i]])

            label_list, frequency_list = myutils.find_frequency_of_each_element_in_list(votes)
            y_predicted.append(label_list[frequency_list.index(max(frequency_list))])

        return y_predicted

class MyDummyClassifier:
    """Represents a "dummy" classifier using the "most_frequent" strategy.
        The most_frequent strategy is a Zero-R classifier, meaning it ignores
        X_train and produces zero "rules" from it. Instead, it only uses
        y_train to see what the most frequent class label is. That is
        always the dummy classifier's prediction, regardless of X_test.

    Attributes:
        most_common_label(obj): whatever the most frequent class label in the
            y_train passed into fit()

    Notes:
        Loosely based on sklearn's DummyClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    """
    def __init__(self):
        """Initializer for DummyClassifier.

        """
        self.most_common_label = None

    def fit(self, X_train, y_train):
        """Fits a dummy classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Zero-R only predicts the most frequent class label, this method
                only saves the most frequent class label.
        """
        label_list, frequencies = myutils.find_frequency_of_each_element_in_list(y_train)

        self.most_common_label = label_list[frequencies.index(max(frequencies))]

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = [self.most_common_label for i in range(len(X_test))]
        return y_predicted 

class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        priors(dict of key:str and value:float(P)): The prior probabilities computed for each
            label in the training set.
        posteriors(list of dict of key:attribute value and value of dict:class and value:float(P)): The posterior probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.priors = {}
        self.posteriors = []

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        # clear for refitting
        self.priors = {}
        self.posteriors = {}

        # set priors -- count the number of each class and divide by total instances
        freq_dict = {}
        for label in y_train:
            if label in freq_dict.keys():
                freq_dict[label] += 1
            else:
                freq_dict[label] = 1
        for key, value in freq_dict.items():
            self.priors[key] = value/len(y_train)

        # set posteriors
        self.posteriors = [{}] * len(X_train[0])

        for col_index in range(len(X_train[0])):
            self.posteriors[col_index] = {}

            # setup index's storage structure
            column = []
            for row_index in range(len(X_train)):
                column.append(X_train[row_index][col_index])
            item_label_list, parallel_frequency_list = myutils.find_frequency_of_each_element_in_list(column)
            for item in item_label_list:
                self.posteriors[col_index][item] = {}
                y_label_list, y_parallel_frequency_list = myutils.find_frequency_of_each_element_in_list(y_train)
                for y_label in y_label_list:
                    self.posteriors[col_index][item][y_label] = 0
                    
            # find number of occurances for each class based on a given label
            for row_index in range(len(X_train)):
                value_at_index = X_train[row_index][col_index]
                class_at_index = y_train[row_index]
                self.posteriors[col_index][value_at_index][class_at_index] += 1
            # divide each total
            for val_key in self.posteriors[col_index].keys():
                for class_key in self.posteriors[col_index][val_key].keys():
                    self.posteriors[col_index][val_key][class_key] /= (self.priors[class_key] * len(y_train))

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        class_labels = list(self.priors.keys())
        for test_item in X_test:
            best_value = 0
            best_index = 0
            for i, class_label in enumerate(class_labels):
                prior_val = self.priors[class_label]
                for j in range(len(test_item)):
                    if test_item[j] in self.posteriors[j]:
                        prior_val *= self.posteriors[j][test_item[j]][class_label]
                if best_value < prior_val:
                    best_value = prior_val
                    best_index = i
            y_predicted.append(class_labels[best_index])
                
        return y_predicted

class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.tree = None

    def fit(self, X_train, y_train, F = None):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
            F(int): size of subset created at each node, can be None in which case 
                use all avalible attributes

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        self.X_train = X_train
        self.y_train = y_train
        train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
        available_attributes = []
        for x in range(len(X_train[0])):
            available_attributes.append(str("att" + str(x)))
        domain = {}
        for x in range(len(X_train[0])):
            seen, count = myutils.get_freq(X_train, x)
            domain[x] = seen
        for type in domain:
            domain[type].sort()
        # also recall that python is pass by object reference
        if F is None:
            self.tree = myutils.tdidt(train, available_attributes, domain)
        else:
            self.tree = myutils.tdidt(train, available_attributes, domain, F)


    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        results = []
        for test in X_test:
            results.append(myutils.recurse_predict(test, self.tree))


        return results

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format
        "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        list = myutils.recurse_rules(self.tree, "", attribute_names, class_name)
        for row in list:
            print(row)

        


    # BONUS method
    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """BONUS: Visualizes a tree via the open source Graphviz graph visualization package and
        its DOT graph language (produces .dot and .pdf files).

        Args:
            dot_fname(str): The name of the .dot output file.
            pdf_fname(str): The name of the .pdf output file generated from the .dot file.
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).

        Notes:
            Graphviz: https://graphviz.org/
            DOT language: https://graphviz.org/doc/info/lang.html
            You will need to install graphviz in the Docker container as shown in class to complete this method.
        """
        print()
        print()
        print("----------------------------------------------")
        dot_file = open(dot_fname, 'w')
        dot_file.write("graph g {\n")
        self.visualize_tree_recurr(attribute_names, dot_file, self.tree)
        dot_file.write("}")
        dot_file.close()

        os.popen("dot -Tpdf -o " + pdf_fname + " " + dot_fname)

    def visualize_tree_recurr(self, attribute_names, dot_file, curr_tree):
        # write current attribute
        # always indent 4 spaces 
        INDENTATION = " " * 4

        current_attribute_name = curr_tree[1] 
        current_attribute_identifier = curr_tree[1] + str(np.random.randint(0, 10000))

        value_subtrees_accessable_from_attribute = curr_tree[2:]

        if attribute_names is not None:
            # convert current attribute to its attribute_name from generic att#
            index_of_attribute = self.get_index_of_attribute(curr_tree[1])
            current_attribute_name = attribute_names[index_of_attribute] 

        dot_file.write(INDENTATION + str(current_attribute_identifier) + " [label=" + str(current_attribute_name) + " shape=box]\n")

        for value in value_subtrees_accessable_from_attribute:
            next_subtree = value[2]

            # base case when leaf is found
            if next_subtree[0] == "Leaf":
                # make a random name to prevent naming collisions
                leaf_name = str(next_subtree[1]) + str(np.random.randint(0, 10000))
                label = str(next_subtree[1]) + " " + str(next_subtree[2]) + "/" + str(next_subtree[3]) # set the label as the value and the percent (frac)
                dot_file.write(INDENTATION + str(leaf_name) + " [label=\"" + str(label) + "\"]\n")

                edge_label = value[1] 
                dot_file.write(INDENTATION + str(current_attribute_identifier) + " -- " + str(leaf_name) + "[label=\"" + str(edge_label) + "\"]\n")
            else:
                backtrack_attribute_name = self.visualize_tree_recurr(attribute_names, dot_file, next_subtree)

                # after recurse, add labeled edge between attribute and next node
                edge_label = value[1]
                
                dot_file.write(INDENTATION + str(current_attribute_identifier) + " -- " + str(backtrack_attribute_name) + "[label=\"" + str(edge_label) + "\"]\n")
        return current_attribute_identifier

    def get_index_of_attribute(self, attribute_name):
        """Because of our attribute naming convention,
        we can look at the end of the string name to find the attributes original index"""
        return int(attribute_name[3:])


class MyRandomForestClassifier():
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        trees(list of MyDescisionTreeClassifiers)

    """
    def __init__(self):
        """Initializer for MyRandomForestClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.trees = None

    def fit(self, X_train, y_train, N, M, F):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
            N(int): the number of "random" descision trees
            F(int): num remaining attributes as candidates to partition on
            M(int): num of most accurate trees to take(M < N)

        Notes:
            
        """
        self.X_train = X_train
        self.y_train = y_train

        # create test and remainder sets
        remainder_X, remainder_y, test_X,  test_y = myevaluation.train_test_split(X_train, y_train, test_size=0.33) 

        # create trees 
        trees = []                      # the list we will be storing the trees for the "forest"
        parallel_accuracy_scores = []   # an accuracy score for each tree in the forest
        for _ in range(N):
             # bootstrap the data
            # create the random stratified test set 
            # consisting of one third of the original data set,
            # with the remaining two thirds of the instances forming the "remainder set".
            training_indices, validation_indices = self.compute_bootstraped_indices(len(remainder_X))

            # convert indices to instances 
            fit_instances = [remainder_X[training_indices[i]] for i in range(len(training_indices))]
            fit_classes = [test_X[training_indices[i]] for i in range(len(training_indices))]
            test_instances = [remainder_X[validation_indices[i]] for i in range(len(validation_indices))]
            test_classes = [test_X[validation_indices[i]] for i in range(len(validation_indices))]

            # fit the tree on the training instances 
            new_tree = MyDecisionTreeClassifier()
            new_tree.fit(fit_instances, fit_classes, F)

            # compare the tree against the test cases
            # the validation set will be used to test
            
            predictions = new_tree.predict(test_instances)
            accuracy = myevaluation.accuracy_score(test_classes, predictions, normalize=True)
            
            parallel_accuracy_scores.append(accuracy)

            trees.append(new_tree)
        
        # find the M best performing trees
        sorted_accuracies = sorted(parallel_accuracy_scores)
        best_scores = sorted_accuracies[:M]
        best_performing_trees = [trees[parallel_accuracy_scores.index(score)] for score in best_scores]
        
        self.trees = best_performing_trees
    
    def compute_bootstraped_indices(self, table_size):
        """finds bootstrap sample, returns a list of indices
            train -> D random indices in table range
            validation -> indices not chosen by random selection"""
        training = []
        validation = []

        for _ in range(table_size):
            rand_index = np.random.randint(0, table_size) # Return random integers from low (inclusive) to high (exclusive)
            training.append(rand_index)
        validation = [i for i in range(table_size) if training.count(i) == 0]

        return training, validation

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        # majority voting 
        # gather all predictions from each of the trees
        prediction_matrix = []  # used to store each prediction from the trees (parallel)
        for tree in self.trees:
            tree_i_predictions = tree.predict(X_test)
            prediction_matrix.append(tree_i_predictions)

        # indexed by col number, 
        # value is the most common element 
        col_votes = []
        if len(self.trees) > 0:
            # make sure there are trees to count 
            for i in range(len(prediction_matrix[0])):
                col_votes.append(prediction_matrix[0][i])

                # find the most frequent element
                most_common_element_count = 0
                elements_in_col = [prediction_matrix[j][i] for j in range(len(prediction_matrix))]
                
                for element in elements_in_col:
                    if elements_in_col.count(element) > most_common_element_count:
                        most_common_element_count = elements_in_col.count(element)
                        col_votes[i] = element

        return col_votes
