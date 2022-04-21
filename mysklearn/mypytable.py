from mysklearn import myutils

import copy
import csv
from tabulate import tabulate

class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure.
        """
        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        return len(self.data), len(self.data[0])

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Notes:
            Raise ValueError on invalid col_identifier
        """
        column_list = []
        try:
            col_index = self.column_names.index(col_identifier) # get the identifier's index 


            for i in range(0, len(self.data)):
                if self.data[i][col_index] == "NA":

                    if include_missing_values:
                        column_list.append(self.data[i][col_index])
                else:
                    column_list.append(self.data[i][col_index])
        except ValueError:
            pass

        return column_list

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        for i in range(0, len(self.data)):
            for j in range(0, len(self.data[i])):
                try: 
                    self.data[i][j] = float(self.data[i][j])
                except ValueError:
                    pass

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.

        Args:
            row_indexes_to_drop(list of int): list of row indexes to remove from the table data.
        """
        new_table = []

        for i in range(0, len(self.data)):
            if row_indexes_to_drop.count(i) == 0:
                # not in indices to drop 
                new_row = []
                for j in range(0, len(self.data[i])):
                    new_row.append(self.data[i][j])
                new_table.append(new_row)

        self.data = copy.deepcopy(new_table)
    
    def drop_col(self, col_to_drop):
        """Remove col from table

        Args:
            col_to_drop(int): col indexes to remove from the table data and header
        """
        try:
            if type(col_to_drop) == str:
                col_index = self.column_names.index(col_to_drop)
            else:
                col_index = col_to_drop

            new_header = [ self.column_names[i] for i in range(len(self.column_names)) if i != col_index]
            new_table = []

            for i, row in enumerate(self.data):
                new_table.append([])
                for j, col in enumerate(row):
                    if j != col_index:
                        new_table[i].append(col)

            self.column_names = copy.deepcopy(new_header)
            self.data = copy.deepcopy(new_table)
        except ValueError:
            pass

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like
                table = MyPyTable().load_from_file(fname)

        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        infile = open(filename, 'r')
        csv_reader = csv.reader(infile)

        self.column_names = csv_reader.__next__()

        for row in csv_reader:
            self.data.append(row)

        self.convert_to_numeric()
        infile.close()

        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """
        outfile = open(filename, 'w')

        # write out header labels
        for i in range(0, len(self.column_names) - 1):
            outfile.write(self.column_names[i] + ",")
        outfile.write(self.column_names[-1] + "\n")

        # write out rest of data 
        for i in range(0, len(self.data)):
            for j in range(0, len(self.data[i]) - 1):
                outfile.write(str(self.data[i][j]) + ",")
            outfile.write(str(self.data[i][-1]) + "\n")
    
        outfile.close()

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns
            list of int: list of indexes of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
                The first instance of a row is not considered a duplicate.
        """
        duplicate_list = []
        
        try:
            # create a list of all encountered rows
            encountered_items = []
            
            # build list of encountered rows from the 
            for i in range(0, len(self.data)):
                encountered_row = []
                for key in key_column_names:
                    col_index = self.column_names.index(key)
                    encountered_row.append(self.data[i][col_index])
                encountered_items.append(encountered_row)

            # itterate backwards through encountered items
            # if the count of a list item is > 1, pop it and add its index to the duplicate list
            i = len(encountered_items) - 1
            while(i > -1):
                if(encountered_items.count(encountered_items[i]) > 1):
                    encountered_items.pop(i)
                    duplicate_list.append(i)
                i -= 1

            duplicate_list.sort()
        except ValueError:
            pass

        return duplicate_list

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        new_table = []
        for i in range(0, len(self.data)):
            new_table.append(self.data[i])
            for j in range(0, len(self.data[i])):
                if self.data[i][j] == "NA":
                    new_table.remove(self.data[i])

        self.data = copy.deepcopy(new_table)

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
            by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        col_average = 0.0
        col_index = self.column_names.index(col_name)
        non_NA_count = 0

        # find average of column
        for i in range(0, len(self.data)):
            if self.data[i][col_index] != "NA":
                col_average += self.data[i][col_index]
                non_NA_count += 1
        col_average = col_average/non_NA_count

        for i in range(0, len(self.data)):
            if self.data[i][col_index] == "NA":
                self.data[i][col_index] = col_average

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.

        Args:
            col_names(list of str): names of the continuous columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]
        """
        new_table_column_names = ["attribute", "min", "max", "mid", "avg", "median"]
        new_table_data = []

        for i in range(0, len(col_names)):
            try:
                new_table_data_row = []

                # attribute
                new_table_data_row.append(col_names[i])

                # convert the column into a 1D list with get_column
                col_list = self.get_column(col_names[i], False)

                # min 
                min_value = min(col_list)
                new_table_data_row.append(min_value)
                # max 
                max_value = max(col_list)
                new_table_data_row.append(max_value)
                # mid
                new_table_data_row.append((min_value + max_value)/2)
                # avg
                average_value = sum(col_list)/len(col_list)
                new_table_data_row.append(average_value)
                # median 
                median_number = 0.0
                col_list.sort()
                if len(col_list) % 2 == 0:
                    median_number = col_list[int((len(col_list) - 1)/2)] + col_list[int((len(col_list) + 1)/2)]
                    median_number /= 2
                else:
                    median_number = col_list[int(len(col_list)/2)]
                new_table_data_row.append(median_number)
                # add new data row to table
                new_table_data.append(new_table_data_row)
            except ValueError:
                pass

        return MyPyTable(column_names=new_table_column_names, data=new_table_data)

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
            with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """

        table_header = []
        table_data = []

        # add all headers from self table 
        table_header = [key for key in self.column_names]
        # then add other headers from other table

        for key in other_table.column_names:
            if table_header.count(key) == 0:
                table_header.append(key)

        for i_data, row_data in enumerate(self.data):
            for i_other, row_other in enumerate(other_table.data):
                if all((row_data[self.column_names.index(key)] == row_other[other_table.column_names.index(key)] for key in key_column_names)):
                    new_row = ["NA" for x in range(len(table_header))]

                    for i in range(len(row_data)):
                        new_row[table_header.index(self.column_names[i])] = row_data[i]

                    for i in range(len(row_other)):
                        new_row[table_header.index(other_table.column_names[i])] = row_other[i]
                    table_data.append(new_row)

        return MyPyTable(table_header, table_data)

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
            other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pad the attributes with missing values with "NA".
        """

        table_header = []

        # add all headers from self table 
        table_header = [key for key in self.column_names]
        # then add other headers from other table
        for key in other_table.column_names:
            if table_header.count(key) == 0:
                table_header.append(key)

        table_data = []
        
        for i_data, row_data in enumerate(self.data):
            found_match = False

            for i_other, row_other in enumerate(other_table.data):
                if all((row_data[self.column_names.index(key)] == row_other[other_table.column_names.index(key)] for key in key_column_names)):
                    found_match = True
                    new_row = ["NA" for x in range(len(table_header))]

                    for i in range(len(row_data)):
                        new_row[table_header.index(self.column_names[i])] = row_data[i]

                    for i in range(len(row_other)):
                        new_row[table_header.index(other_table.column_names[i])] = row_other[i]
                    table_data.append(new_row)
                    
            if not found_match:
                appended_row = ["NA" for x in range(len(table_header))]
                for i in range(len(row_data)):
                    appended_row[table_header.index(self.column_names[i])] = row_data[i] 
                table_data.append(appended_row)
        
        for row_other in other_table.data:
            found_match = False
            for row_new in table_data:
                if all((row_new[table_header.index(key)] == row_other[other_table.column_names.index(key)] for key in key_column_names)):
                    found_match = True

            if not found_match:
                appended_row = ["NA" for x in range(len(table_header))]
                for i in range(len(row_other)):
                    appended_row[table_header.index(other_table.column_names[i])] = row_other[i] 
                table_data.append(appended_row)

        return MyPyTable(table_header, table_data)