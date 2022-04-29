from cmath import log
from random import random
import mysklearn.myutils
import mysklearn.myutils as myutils
import mysklearn.mypytable
from mysklearn.mypytable import MyPyTable 
import mysklearn.myclassifiers
from mysklearn.myclassifiers import MyKNeighborsClassifier, MyDummyClassifier, MyNaiveBayesClassifier, MyDecisionTreeClassifier
import mysklearn.myevaluation
import mysklearn.myevaluation as myevaluation

league_standing_pytable = MyPyTable()
league_standing_pytable.load_from_file("lib/nhl_leaguestandings.csv")


def bag_set(trees):
    num = len(trees)
    rem_len= num*.63
    test_len=num-rem_len
    return 


def rand_tree(avail_attributes, F):
    #be sure that year is not included if necessary
    at_cop=avail_attributes.copy()
    rand_t=[]#random tree
    track=F
    rand_ats=[]
    while track>0:#randomly select attributes
        randselc=range(len(at_cop))
        ran_int=random.choice(randselc)
        rand_ats.append(at_cop[ran_int])#appends the chosen random attribute
        del at_cop[ran_int]
        track -= 1

    for i in range(F):#create the tree by calling select attributes
        given_att=select_attribute(rand_ats)#selects a desired attribute
        del rand_ats[rand_ats.index(given_att)]
        #add given att to the tree
        #decide on how to select the outcome

    return rand_t

def select_attribute(avail_attributes):#selects an attribute from F random attributes 
    #to split on, using entropy, unless avail is less than F
    att_split=[]
    att_col=[]
    att_probs=[]
    entropy=[]
    att_rec=[]
    for i in range(len(avail_attributes)):
        att_col=league_standing_pytable.get_column(avail_attributes[i])
        att_split=set(att_col)
        att_split=list(att_split)
        
        for z in range(len(att_split)):
            att_rec[z]=0 #creates a record to track each occurance of an instance

        for a in range(len(att_col)):
            att_rec[att_split.index(att_col[a])] += 1 #counts occurances

        length=len(att_col)
        for num in range(len(att_rec)):
            att_probs=att_rec[num]/length #gives probablity of each instance
        
        totalE=0
        
        for prob in range(len(att_probs)):
            totalE+=log(att_probs[prob],2)
        entropy[i]= totalE * -1
        #calcs entropy of a given attribue
        # E = -(log(instance1/total)^2 + (log(instance2/total))

    #split based upon lowest entropy of all given attribute
    min_index=0
    for c in range(len(entropy)):
        if(c==0):
            continue
        elif(entropy[min_index]>entropy[c]) and entropy[c] != 0:
            min_index=c

    #returns the attribute to split on
    return avail_attributes[min_index]
