import numpy as np

import math


def find_frequency_of_each_element_in_list(input_list):
    """finds each unique value and the frequency of that value

        Args:
            input_list(list): list of data
        Returns:
            item_label_list(list): all unique values found
            parallel_frequency_list(list): the frequencies of those unique values
    """
    item_label_list = []            # name of the elements that were encountered
    parallel_frequency_list = []    # frequencies of each of the elements

    for item in input_list:
        if item_label_list.count(item) == 0:
            item_label_list.append(item)
            parallel_frequency_list.append(1)
        else:
            existing_item_index = item_label_list.index(item)
            parallel_frequency_list[existing_item_index] += 1

    return item_label_list, parallel_frequency_list

def pretty_print_labels_and_frequencies(labels, frequencies):
    """prints a tables labels and frequencies in a somewhat "pretty" way 
    """
    print("Labels | Frequencies")

    for i in range(len(labels)):
        print(str(labels[i]) + ": " + str(frequencies[i]))

def string_numeric_percentage_to_int(string_percent):
    """converts a string percentage value into an int 

        Args:
            string_percent(str): the string percent value
        Returns:
            output(int): an integer percent value
    """
    i = 0
    output = ""

    # read through any leading spaces
    while i < len(string_percent) and string_percent[i] == ' ':
        i += 1

    while i < len(string_percent) and string_percent[i] != '%' and string_percent[i] != ' ':
        output += str(string_percent[i])
        i += 1

    return int(output)

def normalize_list(list):
    """finds all values in x and y where not "NA" in both 

        Args:
            x(list of float/int): x values
            y(list of float/int): y values
        Returns:
            output_x(list): x cleaned list 
            output_y(list): y cleaned list 
        Note:
            This is intended to find all values which x and y have existant values in common
            to make a graph
    """
    normalized_list = []

    min_val = min(list)
    max_val = max(list)

    for i in range(len(list)):
        normalized_list.append((list[i] - min_val)/(max_val - min_val))

    return normalized_list

def remove_NA_vals(list):
    """finds all values in x and y where not "NA" in both 

        Args:
            x(list of float/int): x values
            y(list of float/int): y values
        Returns:
            output_x(list): x cleaned list 
            output_y(list): y cleaned list 
        Note:
            This is intended to find all values which x and y have existant values in common
            to make a graph
    """
    output_list = []
    for item in list:
        if item != "NA":
            output_list.append(item)
    
    return output_list

def find_all_non_NA_matches(x, y):
    """finds all values in x and y where not "NA" in both 

        Args:
            x(list of float/int): x values
            y(list of float/int): y values
        Returns:
            output_x(list): x cleaned list 
            output_y(list): y cleaned list 
        Note:
            This is intended to find all values which x and y have existant values in common
            to make a graph
    """
    output_x = []
    output_y = []

    for i in range(len(x)):
        if x[i] != "NA" and y[i] != "NA":
            output_x.append(x[i])
            output_y.append(y[i])
    
    return output_x, output_y
    
def compute_slope_intercept(x, y):
    """find correlation coefficient

        Args:
            x(list of float/int): x values
            y(list of float/int): y values
        Returns:
            m(float): slope
            b(float): intercept
    """
    numer = 0
    denom = 0

    avg_x = sum(x)/len(x)
    avg_y = sum(y)/len(y)

    for i in range(len(x)):
        numer += ((x[i] - avg_x) * (y[i] - avg_y))
        denom += ((x[i] - avg_x) ** 2)

    m = numer/denom
    b = avg_y - m * avg_x

    return m, b

def calculate_correlation_coefficient(x, y):
    """find correlation coefficient

        Args:
            x(list of float/int): x values
            y(list of float/int): y values
        Returns:
            r(float): correlation coefficient
    """
    numer = 0
    x_denom = 0
    y_denom = 0

    avg_x = sum(x)/len(x)
    avg_y = sum(y)/len(y)

    for i in range(len(x)):
        numer += ((x[i] - avg_x) * (y[i] - avg_y))
        x_denom += (x[i] - avg_x) ** 2
        y_denom += (y[i] - avg_y) ** 2
    
    r = numer / ((x_denom * y_denom) ** 0.5)

    return r

def calculate_covarience(x, y):
    """find covarience

        Args:
            x(list of float/int): x values
            y(list of float/int): y values
        Returns:
            cov(float): covarience
    """
    numer = 0
    denom = 0

    avg_x = sum(x)/len(x)
    avg_y = sum(y)/len(y)

    denom = len(x)

    for i in range(len(x)):
        numer += ((x[i] - avg_x) * (y[i] - avg_y))
    
    cov = numer / denom

    return cov

def compute_equal_width_cutoffs(values, num_bins):
    """function from U3-Data-Analysis utils.py, finds equal length bins in a list of data
    """
    values_range = math.ceil(max(values)) - math.floor(min(values))
    bin_width = values_range / num_bins # float
    # since bin_width is a float, we shouldn't use range() to generate a list
    # of cutoffs, use np.arange()
    cutoffs = list(np.arange(min(values), max(values), bin_width))
    cutoffs.append(max(values)) # exactly the max(values)
    # to handle round off error... 
    # if your application allows, we should convert to int
    # or optionally round them
    cutoffs = [round(cutoff, 2) for cutoff in cutoffs]
    return cutoffs 

# -------------------------------------------------
#   pa4 utils
# -------------------------------------------------

def find_frequency_of_each_element_in_list(input_list):
    """finds each unique value and the frequency of that value

        Args:
            input_list(list): list of data
        Returns:
            item_label_list(list): all unique values found
            parallel_frequency_list(list): the frequencies of those unique values
    """
    item_label_list = []            # name of the elements that were encountered
    parallel_frequency_list = []    # frequencies of each of the elements

    for item in input_list:
        if item_label_list.count(item) == 0:
            item_label_list.append(item)
            parallel_frequency_list.append(1)
        else:
            existing_item_index = item_label_list.index(item)
            parallel_frequency_list[existing_item_index] += 1

    return item_label_list, parallel_frequency_list

def distance(train_instance, test_instance):
    distance = 0

    for i in range(len(train_instance)):
        distance += (train_instance[i] - test_instance[i])**2
    
    return distance**0.5

def print_class_and_actual(instance, classification, actual):
    print("instance: " + str(instance))
    print("class: " + str(classification) + " actual: " + str(actual))

def accuracy(predictions, actual):
    total_matches = 0

    for i in range(len(predictions)):
        if predictions[i] == actual[i]:
            total_matches += 1
    
    return total_matches/len(predictions)

def cal_error_rate(predictions, actual):
    total_matches = 0

    for i in range(len(predictions)):
        if predictions[i] == actual[i]:
            total_matches += 1
    
    return (1 - total_matches/len(predictions)) * 100
    

# -------------------------------------------------
#   pa5 utils
# -------------------------------------------------

def randomized_index_list(n, seed):
    if seed is not None:
        np.random.seed(seed + 1)
    else:
        np.random.seed(0)
        
    shuffled_index_list  = [i for i in range(n)]

    np.random.shuffle(shuffled_index_list)

    return shuffled_index_list

def randomized_index_list_with_replacement(n, seed):
    if seed is not None:
        np.random.seed(seed + 1)
    else:
        np.random.seed(0)
        
    shuffled_index_list  = [np.random.randint(0, high=n) for i in range(n)]

    return shuffled_index_list

def truncate_list(list, n):
    """
    DESC: Reduce a list to size n, remove any elements above n"""

    truncated_list = []
    if n > len(list):
        return list
    if n <= 0:
        return list 

    for i in range(n):
        truncated_list.append(list[i])
    return truncated_list

def combine_lists(list_of_lists, exclusion_list=[]):
    """
    DESC: concatonate two lists"""
    concatenated_list = []

    for i, list in enumerate(list_of_lists):
        if exclusion_list.count(i) == 0:
            for item in list:
                concatenated_list.append(item)

    return concatenated_list

def get_column(table, header, col_name):
    """get_column function from class"""
    col_index = header.index(col_name)
    col = []
    for row in table:
        value = row[col_index]
        if value != "NA":
            col.append(value)
    return col 

def group_by(table, header, groupby_col_name):
    """group_by function from class"""
    groupby_col_index = header.index(groupby_col_name) 
    groupby_col = get_column(table, header, groupby_col_name)
    group_names = sorted(list(set(groupby_col))) 
    group_subtables = [[] for _ in group_names] 
    
    for row in table:
        groupby_val = row[groupby_col_index]
        groupby_val_subtable_index = group_names.index(groupby_val)
        group_subtables[groupby_val_subtable_index].append(row.copy())
    return group_names, group_subtables

def unique_vals(list):
    """Like groupby but using a list sorts 
        list_of_sorting_lists(list of list of int val)
        group_by_list(list of val)"""
    seen_group_by_vals = []
    parallel_numbered_vals = []

    for i in range(len(list)):
        if seen_group_by_vals.count(list[i]) == 0:
            seen_group_by_vals.append(list[i])
            parallel_numbered_vals.append(len(seen_group_by_vals) - 1)
        else:
            parallel_numbered_vals.append(seen_group_by_vals.index(list[i]))

    return seen_group_by_vals, parallel_numbered_vals


# -------------------------------------------------
#   pa7 utils
# -------------------------------------------------
def find_indices_of_all_instance_types(label_list):
    """creates a list of all labels and a parallel list of indexes they were encountered at"""
    labels = []
    parallel_label_indices = []

    for i in range(len(label_list)):
        if labels.count(label_list[i]) == 0:
            labels.append(label_list[i])
            parallel_label_indices.append([])
            parallel_label_indices[len(parallel_label_indices)-1].append(i)
        else:
            index_of_label = labels.index(label_list[i])
            parallel_label_indices[index_of_label].append(i)
    return labels, parallel_label_indices

def indices_in_common(index_list_1, index_list_2):
    """takes two index lists and finds which indices they have in common (assumes no repeated indices in each list)"""
    indices_in_common = 0

    for index in index_list_1:
        if index_list_2.count(index) > 0:
            indices_in_common += 1
    
    return indices_in_common


def indices_list_to_instances(list_of_index_groups, data):
    """This function will convert a list of indices parallel to a 2-D dataset to a list of the data
        This is mainly for algorithms like stratified K-fold, converting returned indices to intances"""
    
    # list of indices should be in format list of list of int
    obj_data_table = []
    for list_of_indices in list_of_index_groups:
        for index in list_of_indices:
            obj_data_table.append(data[index])
    return obj_data_table


def find_count_and_index_positions_of_each_col_items(table_without_labels):
    """
    for each column, find the list of values and each index they can be found at
        list of dictionary of value(key) -> list of indices(instance number)
    NOTE: this will return the columns in the same order they were passed in"""
    list_of_value_index_dict = []

    if len(table_without_labels) > 0:
        for i in range(0, len(table_without_labels[0])):
            # itterate through each col
            value_index_dict = {}

            for j in range(0, len(table_without_labels)):
                # itterate through each instance
                current_value_in_instance = table_without_labels[j][i]
                if list(value_index_dict.keys()).count(current_value_in_instance) == 0:
                    # if new value encountered, add a new key and list
                    new_index_list = [j]
                    value_index_dict[current_value_in_instance] = new_index_list
                else:
                    # otherwise add the current index to its value
                    value_index_dict[current_value_in_instance].append(j)
            
            list_of_value_index_dict.append(value_index_dict)
    
    return list_of_value_index_dict
    

## Descision Tree Util Functions


def randomize_in_place(alist, parallel_list=None):
    for i in range(len(alist)):
        # generate a random index to swap this value at i with
        rand_index = np.random.randint(0, len(alist)) # rand int in [0, len(alist))
        # do the swap
        alist[i], alist[rand_index] = alist[rand_index], alist[i]
        if parallel_list is not None:
            parallel_list[i], parallel_list[rand_index] =\
                parallel_list[rand_index], parallel_list[i]

def seed(seed):
    np.random.seed(seed)

def random_number(upper, lower):
    return np.random.randint(lower,upper)

def freq(y_train):
    """

    Args:
        y_train: the table that needs the freq checked

    Returns: the most freq val

    """
    sorted_list = sorted(y_train)  # inplace
    # parallel lists
    values = []
    counts = []
    for value in sorted_list:
        if value in values:  # seen it before
            counts[-1] += 1  # okay because sorted
        else:  # haven't seen it before
            values.append(value)
            counts.append(1)
    max = 0
    max_val = ""
    for x in range(len(counts)):
        if counts[x] > max:
            max = counts[x]
            max_val = values[x]
    return max_val


def get_freq(labels, index=None):
    if index is None:
        sorted_labels = sorted(labels)
        counts = []
        seen = []
        for val in sorted_labels:
            if val in seen:
                counts[-1] += 1
            else:
                seen.append(val)
                counts.append(1)
        return seen, counts
    else:
        counts = []
        seen = []
        for val in labels:
            if val[index] in seen:
                counts[seen.index(val[index])] += 1
            else:
                seen.append(val[index])
                counts.append(1)
        return seen, counts

def likelihood_table(index, X, y):
    # index = current label to make table for
    # X = current X_train set
    # y = current y_train set
    seen, counts = get_freq(y) # gets the amount of different result labels

    att_list = [] # att list is the listings of the att value and the result at that value


    # gets all attributes within the column and matches them with y train
    for row in range(len(X)):
        cur_row = X[row]
        att = [cur_row[index], y[row]]
        att_list.append(att)

    att_list_labeled = []

    seen_label = []
    for row in att_list:
        if row[0] in seen_label:
            att_list_labeled[seen_label.index(row[0])].append(row)
        else:
            att_list_labeled.append([row])
            seen_label.append(row[0])
    # sorted into each label
    # now break into number of each result
    result_header = seen

    table = []
    for row in att_list_labeled:
        new_row = [row[0][0]]
        for x in range(len(result_header)):
            new_row.append(0)
        for att in row:
            idx = result_header.index(att[-1])
            new_row[idx+1] += 1
        table.append(new_row)
    result_header.insert(0, "Label")
    # stitched table based on result returned wit the header
    return table, result_header

def calculate_post(table, label, table_header):
    # given a liklihood table as defined in liklihood_table()
    post = []
    dict = {}
    for x in range(len(table_header)):
        top = 0
        bottom = 0
        if x != 0:
            for row in table:
                if row[0] == label:
                    top += row[x]
                    bottom += row[x]
                else:
                    bottom += row[x]
            dict[table_header[x]] = top/bottom
            post.append(dict)
        else:
            post.append(label)
    return post


def full_post_calc(X, y):
    posts = {}
    for index in range(len(X[0])):
        table, header = likelihood_table(index, X, y)
        strn = "att" + str(index)
        postpeice = {}
        for row in table:
            post = calculate_post(table, row[0], header)
            label = post.pop(0)
            label_dict = post.pop(0)
            postpeice[label] = label_dict
        posts[strn] = postpeice
    return posts

def compute_euclidean_distance(v1, v2):
    """

    Args:
        v1: vector 1
        v2: vector 2

    Returns: the euclidian distance

    """
    if v1 == v2:
        return 0
    else:
        return 1

def tdidt(current_instances, available_attributes, domain):
    # basic approach (uses recursion!!):
    #print("available_attributes:", available_attributes)

    # select an attribute to split on
    attribute = select_attribute(current_instances, available_attributes)
    #print("splitting on attribute:", attribute)
    available_attributes.remove("att" + str(attribute))
    tree = ["Attribute", str("att" +str(attribute))]
    # group data by attribute domains (creates pairwise disjoint partitions)
    partitions = partition_instances(current_instances, attribute, domain)
    # for each partition, repeat unless one of the following occurs (base case)
    for att_value, att_partition in partitions.items():
        #print("")
        #print("current attribute value:", att_value, len(att_partition))
        value_subtree = ["Value", att_value]
        #    CASE 1: all class labels of the partition are the same => make a leaf node
        if len(att_partition) > 0 and all_same_class(att_partition): # returns true if all same, false otherwise
            #print("Case 1")
            denom = 0
            for val in partitions:
                denom += len(partitions[val])
            value_subtree.append(["Leaf", att_partition[0][-1], len(att_partition), denom])
            tree.append(value_subtree)
        #    CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
        elif len(att_partition) > 0 and len(available_attributes) == 0:
            #print("Case 2")
            seen, count = get_freq(att_partition, -1)
            highest = max(count)
            index = count.index(highest)
            denom = 0
            for val in partitions:
                denom += len(partitions[val])
            subtree = ["Leaf", seen[index], denom]
            value_subtree.append(subtree)
            tree.append(value_subtree)

        #    CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
        elif len(att_partition) == 0:
            #print("Case 3")
            value_subtree = ["Leaf"]
            return value_subtree
            # TODO: backtrack to replace the attribute node with majority vote
        else:
            # recursive step
            subtree = tdidt(att_partition, available_attributes.copy(), domain)
            # i built both case 3 and 2 into the recursive part because it does the backtracking stuff
            # and this is how i was taught to do this stuff
            # plus it works. if it aint broke dont fix it yeah?

            if subtree[0] == "Leaf": # case 3 handler
                seen, count = get_freq(att_partition,-1)
                highest = max(count)
                index = count.index(highest)
                denom = 0
                for val in partitions:
                    denom += len(partitions[val])
                subtree = ["Leaf", seen[index], len(att_partition), denom]

            value_subtree.append(subtree)

            tree.append(value_subtree)

    return tree


def case2_helper(partitions):
    seen, counts = get_freq(partitions, -1)
    max = 0
    index = 0
    sum = 0
    for x in range(len(counts)):
        if counts[x] > max:
            max = counts[x]
            index = x
        sum += counts[x]
    return seen[x], counts[x], sum

def all_same_class(partition):
    label = partition[0][-1]
    for row in partition:
        if row[-1] != label:
            return False
    return True

def select_attribute(instances, attributes):
    #print(instances)
    enew_table = []
    results, counts = get_freq(instances, -1)
    for attribute in attributes:
        partition = {}
        for choice in results:
            partition[choice] = []
        for row in instances:
            partition[row[-1]].append(row)
        seen, counts = get_freq(instances, int(attribute[3:]))
        attenw = 0
        for att in seen:
            totals = []
            for result in results:
                count = 0
                for row in partition[result]:
                    if row[int(attribute[3:])] == att:
                        count += 1
                totals.append(count)
            total = 0
            for val in totals:
                total += val
            enw = 0.0
            for val in totals:
                frac = val/total
                if frac != 0:
                    enw += ((val/total) * math.log2(val/total))

            enw *= -1
            enw *= total/len(instances)
            attenw += enw
        enew_table.append([int(attribute[3:]),attenw])
    #print(enew_table)
    min = enew_table[0]
    for piece in enew_table:
        if piece[-1] < min[-1]:
            min = piece

    return min[0]


def partition_instances(instances, split_attribute, domain):
    # lets use a dictionary
    partitions = {} # key (string): value (subtable)
    att_index = split_attribute # e.g. 0 for level
    att_domain = domain[split_attribute]  # e.g. ["Junior", "Mid", "Senior"]
    for att_value in att_domain:
        partitions[att_value] = []
        for instance in instances:
            if instance[att_index] == att_value:
                partitions[att_value].append(instance)

    return partitions

def recurse_predict(test, cur_tree):
    if cur_tree[0] == "Attribute":  # checking if the attribute row
        attribute = int(cur_tree[1][3])
        test_att = test[attribute]
        for piece in cur_tree:
            if piece[1] == test_att:
                return recurse_predict(test, piece[2])
    elif cur_tree[0] == "Leaf":
        return cur_tree[1]

def recurse_rules(cur_tree, rule_start, attribute_names, class_label):
    rule_list = []
    if cur_tree[0] == "Attribute":  # checking if the attribute row
        attribute = cur_tree[1]
        for piece in cur_tree:
            if type(piece) == list:
                if attribute_names is not None:
                    if len(rule_start) == 0:
                        new_rule_start = "IF " + attribute_names[int(attribute[3])] + "=" + piece[1]
                    else:
                        new_rule_start = rule_start + " AND " + attribute_names[int(attribute[3])] + "=" + piece[1]
                else:
                    if len(rule_start) == 0:
                        new_rule_start = "IF " + attribute + "=" + piece[1]
                    else:
                        new_rule_start = rule_start + " AND " + attribute + "=" + piece[1]
                returned_list = recurse_rules(piece[2], new_rule_start, attribute_names, class_label)
                for val in returned_list:
                    rule_list.append(val)

    elif cur_tree[0] == "Leaf":
        rule_start += " THEN " + class_label + "=" + cur_tree[1]
        rule_list.append(rule_start)
        return rule_list
    return rule_list


def seed(seed):
    np.random.seed(seed)