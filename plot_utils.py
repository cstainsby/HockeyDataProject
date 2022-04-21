import matplotlib.pyplot as plt

import mysklearn.myutils as myutils

# -------------------------------------------------------------------------------------
# graph creation functions
# -------------------------------------------------------------------------------------

def scatter_plot_with_descent_line(dataset_titles, dataset, x_col_name, y_col_name, group_by_class_label=None):
    """create a scatter plot with regression line and find correlation coefficient as well as covarience

        Args:
            dataset_titles(list(str)): list of labels for data being graphed
            dataset(list(list())): data being graphed
            x_col_name(str): name of data used for x
            y_col_name(str): name of data used for y
            group_by_class_label(str): name of col with class 
        Returns:
            None
    """
    if group_by_class_label is None :
        print("test")
        print("titles: ", dataset_titles)
        x_index = dataset_titles.index(x_col_name)
        y_index = dataset_titles.index(y_col_name)
        print("x_index: ", str(x_index))
        print("y_index: ", str(y_index))

        cleaned_x, cleaned_y = myutils.find_all_non_NA_matches(dataset[x_index], dataset[y_index])
        m, b = myutils.compute_slope_intercept(cleaned_x, cleaned_y)

        corre_coeff = myutils.calculate_correlation_coefficient(cleaned_x, cleaned_y)
        covarience = myutils.calculate_covarience(cleaned_x, cleaned_y)

        plt.figure(figsize=(10,5))
        plt.title(x_col_name + " vs " + y_col_name)
        plt.ylabel(y_col_name)
        plt.xlabel(x_col_name)
        plt.scatter(cleaned_x, cleaned_y)
        plt.plot([min(cleaned_x), max(cleaned_x)], [m * min(cleaned_x) + b, m * max(cleaned_x) + b], c="r")
        plt.grid(True)
        plt.show()
    else:
        group_names, group_subtables = myutils.group_by(dataset, dataset_titles, group_by_class_label)
        print(group_names)
        print("group subtables: ", group_subtables)


    print("Correlation Coefficient: " + str(corre_coeff) + "\nCovarience: " + str(covarience))
