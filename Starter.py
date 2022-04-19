import csv
import random

#For this assignment we need to make 3 working classifiers
#and have a good daaset and direction

def import_data():
    #import the csv and table
    data= open(hockeydata.csv)
    #replace with file name

    reader=csv.reader(data)
    header=next(reader)
    table=[]
    for row in reader:
        table.append(row)
    data.close()
    return table, header

def generate_test(table, header):
    #generates a random stratified test set
    return 0

def naiveb_classifier(table, header):
    #Uses Naive Bayes to Classify data

    return 0

def decision_tree(table, header):
    #Uses K nearest neighbors to classify a given tree
    attributes= header

    return 0

def rand_tree(attributes):
    at_cop=attributes.copy()
    rand_t=[]#random tree
    counta=len(attributes)#length of attributes
    countt = 0
    while counta>0:
        randselc= range(len(counta))#gets attributes 0 to attribute
        # length as a list
        ran_int=random.choice(randselc)#chooses a random number
        
        #equate that random number to an attribute
        #make it the current part of the list
        #make selections based off of it

        rand_t[countt]=at_cop[counta]
        del at_cop[counta]

        #inputs a given attribue, needs to be altered to have proper
        #indentaion for trees.

        counta -= 1
        countt += 1

    return rand_t

def random_forest_classifier(table, header):
    #creates random trees via selecting a 
    #random next attribute and tests them all over the 
    #remainder set
    #uses rand tree for random tree production
    return 0

