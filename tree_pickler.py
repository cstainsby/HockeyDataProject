import pickle 

import hockey_data_utils
from mysklearn.mypytable import MyPyTable
from mysklearn.myclassifiers import MyDecisionTreeClassifier, MyRandomForestClassifier

# package up classifiers for app to use
valid_classifiers = ["rand_forest", "descision_tree"] # NOTE: Dont change list

print("Valid Classifieres Include:")
for i, classifier_name in enumerate(valid_classifiers):
  print(i, ": " + classifier_name)
chosen_classifier_index = input("Input index of classifier to pickle(0-" + str(len(valid_classifiers)-1) + "): ")

chosen_classifier_name = valid_classifiers[int(chosen_classifier_index)]

# get data from csv
hockey_pytable = MyPyTable()
hockey_pytable.load_from_file("lib/nhl_leaguestandings.csv")
hockey_data_utils.drop_unused_cols(hockey_pytable)

X_train, y_train = hockey_data_utils.get_X_and_y_data(hockey_pytable)

# names of the attributes from the initial data set we 
#     will be using for classification
accepted_attribute_names = hockey_pytable.column_names

# if random forest this will be a list of tree solution 
#   as opposed to tree's single tree solution
classifier_solution = []
if chosen_classifier_name == "rand_forest":
  # tune parameters for forest (N, M, F)
  N = input("(N) How many initial weak trees: ")
  M = input("(M) How many derived strong trees(N>=M): ")
  F = input("(F) Random attributes avalible at each tree split: ")

  if (N < 0) or (M < 0) or (F < 0):
    print("N, M, and/or F set to negative number")
    exit()
  if N < M: 
    print("")
    exit()

  forest_clf = MyRandomForestClassifier()
  forest_clf.fit(X_train, y_train, N, M, F)
  classifier_solution = forest_clf.trees

elif chosen_classifier_name == "descision_tree":
  F = input("(F) Random attributes avalible at each tree split: ")

  if F < 0:
    print("N, M, and/or F set to negative number")
    exit()

  tree_clf = MyDecisionTreeClassifier()
  tree_clf.fit(X_train, y_train, F)
  classifier_solution = tree_clf.tree

else:
  print("Error: Invalid classifier Called")

# the packaged object should always be in the form 
#     [type, header:(list of str), packaged classifier solution]
packaged_object = [chosen_classifier_name, accepted_attribute_names, classifier_solution]
outfile = open("thePickleZone/tree.p", "wb")
pickle.dump(packaged_object, outfile)
outfile.close()