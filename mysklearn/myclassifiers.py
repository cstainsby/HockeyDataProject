import math
import operator
import numpy as np
import os

from mysklearn import myutils
from mysklearn.mysimplelinearregressor import MySimpleLinearRegressor

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
                d = myutils.distance(train_instance, test_instance)
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

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

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
        self.tree = myutils.tdidt(train, available_attributes, domain)


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
        print()
        print()
        print("----------------------------------------------")
        print("rules:")
        curr_string = ["IF "]
        self.print_descision_tree_recurr(attribute_names, self.tree, curr_string, class_name)

    def print_descision_tree_recurr(self, attribute_names, curr_tree, curr_string, class_name=None):

        if curr_tree[0] != "Leaf":
            if attribute_names is not None:
                attribute_name = attribute_names[self.get_index_of_attribute(curr_tree[1])]
                curr_string.append(attribute_name)
            else:
                curr_string.append(curr_tree[1])

            value_subtrees = curr_tree[2:] # all values edges reachable from current attribute
            for value_subtree in value_subtrees:
                if value_subtree[2][0] == "Leaf":
                    curr_string.append(" THEN ")
                else:
                    curr_string.append(" AND ")
                
                self.print_descision_tree_recurr(attribute_names, value_subtree[2], curr_string)

        else:
            out_string = ""
            for string in curr_string:
                out_string += str(string)
            if class_name is None:
                print(out_string + "class = " + str(curr_tree[1]))
            else:
                print(out_string + class_name + "= " + str(curr_tree[1]))

        


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

        dot_file.write(INDENTATION + current_attribute_identifier + " [label=" + current_attribute_name + " shape=box]\n")

        for value in value_subtrees_accessable_from_attribute:
            next_subtree = value[2]

            # base case when leaf is found
            if next_subtree[0] == "Leaf":
                # make a random name to prevent naming collisions
                leaf_name = next_subtree[1] + str(np.random.randint(0, 10000))
                label = next_subtree[1] + " " + str(next_subtree[2]) + "/" + str(next_subtree[3]) # set the label as the value and the percent (frac)
                dot_file.write(INDENTATION + leaf_name + " [label=\"" + label + "\"]\n")

                edge_label = value[1] 
                dot_file.write(INDENTATION + current_attribute_identifier + " -- " + leaf_name + "[label=\"" + edge_label + "\"]\n")
            else:
                backtrack_attribute_name = self.visualize_tree_recurr(attribute_names, dot_file, next_subtree)

                # after recurse, add labeled edge between attribute and next node
                edge_label = value[1]
                
                dot_file.write(INDENTATION + current_attribute_identifier + " -- " + backtrack_attribute_name + "[label=\"" + edge_label + "\"]\n")
        return current_attribute_identifier

    def get_index_of_attribute(self, attribute_name):
        """Because of our attribute naming convention,
        we can look at the end of the string name to find the attributes original index"""
        return int(attribute_name[3:])