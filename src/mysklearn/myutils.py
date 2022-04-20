import numpy as np

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
    
    
