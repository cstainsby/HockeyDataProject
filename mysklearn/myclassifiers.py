import math
import operator
import numpy as np
import graphviz 
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
        # TODO: programmatically extract the header (e.g. ["att0", 
        # "att1", ...])
        # and extract the attribute domains
        header = ["att" + str(i) for i in range(len(X_train[0]))]
        # now, I advise stitching X_train and y_train together
        train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
        # next, make a copy of your header... tdidt() is going
        # to modify the list
        available_attributes = header.copy()
        # also: recall that python is pass by object reference
        self.tree = self.tdidt(train, available_attributes)
        # note: unit test is going to assert that tree == interview_tree_solution
        # (mind the attribute domain ordering)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for test_instance in X_test:
            position = self.tree

            while position[0] != "Leaf":
                # at the current position find the attribute being split on
                #   based on what the value is move down the branch which matches
                attribute_being_split_on = position[1]
                index_of_att = self.get_index_of_attribute(attribute_being_split_on)
                value_of_instance = test_instance[index_of_att]

                branches = position[2:]
                branch_values = [branch[1] for branch in branches]

                index_of_next_position = branch_values.index(value_of_instance)
                position = position[index_of_next_position + 2][2]

            y_predicted.append(position[1])
        return y_predicted

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

    # ----------------------------------------------------------------------------------
    #   tdidt and tdidt helper functions
    # ----------------------------------------------------------------------------------
    def tdidt(self, current_instances,  available_attributes):
        """
        current_instances: table, with class appended, the current partition
        avalible_attributes: column headers that are still avalible, parallel to current_instances
        class_index: index of class col.
        """
        # basic approach (uses recursion!!):
        print()
        print()
        print("-------------------Start-------------------")
        print("available_attributes:", available_attributes)

        # select an attribute to split on
        attribute = self.select_attribute(current_instances, available_attributes)
        available_attributes.remove(attribute)
        tree = ["Attribute", attribute]

        # group data by attribute domains (creates pairwise disjoint partitions)
        attribute_domains_and_occurance = self.find_domain_of_attribute(current_instances, attribute)
        partitions = self.partition_instances(available_attributes, attribute_domains_and_occurance.keys(), current_instances, attribute)

        # for each partition, repeat unless one of the following occurs (base case)
        for att_value, att_partition in partitions.items():
            print("tree: ", self.tree)
            print("curent attribute value:", att_value, len(att_partition))
            value_subtree = ["Value", att_value]
            #    CASE 1: all class labels of the partition are the same => make a leaf node
            if len(att_partition) > 0 and self.all_same_class(att_partition):
                print("CASE 1 all same class")
                # TODO: make a leaf node
                val_of_leaf =  att_partition[0][-1] # class
                num_total_intances_at_level = len(current_instances)
                num_in_leaf = len(att_partition)
                leaf_node = ["Leaf", val_of_leaf,  num_in_leaf, num_total_intances_at_level]
                value_subtree.append(leaf_node)

            #    CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
            elif len(att_partition) > 0 and len(available_attributes) == 0:
                print("CASE 2 no more attributes")
                # TODO: we have a mix of labels, handle clash with majority
                # vote leaf node
                print("partition: ", att_partition)

            #    CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
            elif len(att_partition) == 0:
                print("CASE 3 empty partition")
                # TODO: "backtrack" to replace the attribute node
                # with a majority vote leaf node




            else: # the previous conditions are all false... recurse!!
                subtree = self.tdidt(att_partition, available_attributes.copy())
                # note the copy
                # TODO: append subtree to value_subtree and to tree
                # appropriately
                value_subtree.append(subtree) # append next attribute after split
            tree.append(value_subtree) # append values being split on 

        return tree

    def select_attribute(self, current_instances, availible_attributes):
        """
        current_instances: the current partition
        avalible_attributes:
        """
        # TODO: use entropy to compute and choose the attribute
        # with the smallest Enew
        # for now, we will just choose randomly
        # return list parallel to X_train with the entropy of each col
        entropy_vals = []
        total_labels = len(current_instances)


        list_of_value_index_dict = myutils.find_count_and_index_positions_of_each_col_items(current_instances)
        # remove dictionaries that shouldnt be considered 
        list_of_indices_that_should_be_considered =\
             [self.get_index_of_attribute(attribute) for attribute in availible_attributes]
        list_of_indices_that_should_be_removed =\
            [i for i in range(len(availible_attributes)) if list_of_indices_that_should_be_considered.count(i) == 0]
        for i in range(len(current_instances[0])-1, -1, -1):
            if list_of_indices_that_should_be_removed.count(i) > 0:
                list_of_value_index_dict.pop(i)


        # the last index in the dict will be reserved for class labels
        class_label_dict = list_of_value_index_dict[len(list_of_value_index_dict)-1]
        keys_in_class_dict = class_label_dict.keys()

        for i in range(0, len(list_of_value_index_dict) - 1):
            # for each attribute we are going to find the entropy 
            summation_term = 0

            dict_of_attribute_indices = list_of_value_index_dict[i]
            # find all unique classes avalible to each attribute
            
            keys_in_attribute_dict = list(dict_of_attribute_indices.keys())
            for key in keys_in_attribute_dict:
                num_instances_with_attribute_val = len(dict_of_attribute_indices[key])


                # we need to total the ammount of the instances we are working with
                #   are under each class label
                for class_key in keys_in_class_dict:
                    # calculate each summation term
                    class_val_index_list = class_label_dict[class_key]
                    attribute_val_index_list = dict_of_attribute_indices[key]
                    num_indicies_in_common = myutils.indices_in_common(class_val_index_list, attribute_val_index_list)

                    frac = num_indicies_in_common/num_instances_with_attribute_val
                    
                    if frac != 0:
                        summation_term += -frac * math.log(frac, 2)

            entropy_vals.append((num_instances_with_attribute_val/total_labels) * summation_term)

        index_with_lowest_Enew = 0
        lowest_entropy = entropy_vals[0]
        print("Entropy vals: ", entropy_vals)
        for i, entropy_val in enumerate(entropy_vals):
            if entropy_val < lowest_entropy:
                index_with_lowest_Enew = i
        return availible_attributes[index_with_lowest_Enew]

    def find_domain_of_attribute(self, current_instances, attribute):
        """finds the attribute domains at the current level"""
        domain_and_num_instances = {} # domain(option) : num occurences(int)
        index_of_attribute = self.get_index_of_attribute(attribute)
            
        for instance in current_instances:
            # if unique value found, add it to domain list
            if list(domain_and_num_instances.keys()).count(instance[index_of_attribute]) == 0:
                domain_and_num_instances[instance[index_of_attribute]] = 1
            else:
                domain_and_num_instances[instance[index_of_attribute]] += 1
        return domain_and_num_instances


    def get_index_of_attribute(self, attribute_name):
        """Because of our attribute naming convention, 
        we can look at the end of the string name to find the attributes original index"""
        return int(attribute_name[3:])
            
    
    def partition_instances(self, available_attributes, attribute_domains, instances, split_attribute):
        """group instances by the split attribute domains"""
        # lets use a dictionary
        partitions = {} # key (string): value (subtable)
        att_index = self.get_index_of_attribute(split_attribute) # e.g. 0 for level
        # att_domain = attribute_domains[att_index] # e.g. ["Junior", "Mid", "Senior"]
        
        for att_value in attribute_domains:
            partitions[att_value] = []
            
            # loop through the current instances and for each value 
            #   add it to its partition based on split attribute
            for instance in instances:
                if instance[att_index] == att_value:
                    partitions[att_value].append(instance)

        return partitions
    
    def all_same_class(self, partition_data):
        """checks a partition to see if all values within it are the same class
            partition_data: instances that are appended to the current partition"""
        # the class attribute will always be appended to the end of the data
        first_instance_class = ""
        for i, instance in enumerate(partition_data):
            if i == 0:
                first_instance_class = instance[-1]
            else:
                if instance[-1] != first_instance_class:
                    return False
        return True


    
