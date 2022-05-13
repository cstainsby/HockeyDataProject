from mysklearn.mypytable import MyPyTable
"""
This file is used for keeping cleaning consitant across the project
contains various utility functions which achieve that

This file will work with lib/nhl_leaguestandings.csv specifically
"""

def drop_unused_cols(nhl_pytable):
  """
  Drops a collection of attributes in nhl pytable which are unused when training
  """
  if type(nhl_pytable) != MyPyTable():
     return 

  cols_to_drop = [
    "ROW",
    "RPt%",
    "OL",
    "T",
    "SEASON_TM",
    "CONFERENCE",
    "DIVISION",
    "TEAM",
    "TM",
    "PLAYOFFS"
  ]
  for col_name in cols_to_drop:
    nhl_pytable.drop_col(col_name)

def get_X_and_y_data(nhl_pytable):
  """
  grabs the column that will be used for classification out of the dataset and drops it
  returns the resulting dataset alongside the class col
  """
  if type(nhl_pytable) != MyPyTable():
     return 

  class_col_name = "FINISH"
  y = nhl_pytable.get_column(class_col_name)
  nhl_pytable.drop_col(class_col_name)
  X = nhl_pytable.data

  return X, y