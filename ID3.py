import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
#import time



# from AbstractAlgorithm import AbstractAlgorithm

######### important thoughts ##############

# we dont have empty leaf problem
# we uses all features in every step to calc max_ig
M_values = [1, 2, 3, 5, 8, 16, 30, 50, 80, 120]
train_group = pd.read_csv(r"C:\Users\HP\PycharmProjects\pythonProject\hw3\train.csv")
test_group = pd.read_csv(r"C:\Users\HP\PycharmProjects\pythonProject\hw3\test.csv")
features_num = len(train_group.columns)
features= train_group.columns
train_row_indices= list(range(1,len(train_group)+1))
test_row_indices= list(range(1,len(test_group)+1))

class Node:
    def __init__(self, label=None, feature=None, barrier=None, l_son=None, r_son=None):
        self.label = label
        self.feature = feature
        self.barrier = barrier
        self.l_son = l_son
        self.r_son = r_son


class ID3:
    def __init__(self, limit, is_early_pruning):
        self.decision_tree = None
        self.is_early_pruning= is_early_pruning
        self.limit = limit

    def ig(self, f, examples_indices, examples_entropy):
        filtered_df = train_group.filter(examples_indices, axis=0)
        sorted_examples = (filtered_df[f].sort_values()).reset_index(drop=True)
        examples_size = len(sorted_examples)

        best_barrier = 0.5 * (sorted_examples[0] + sorted_examples[1])

        best_ig = -np.inf
        for i in range(filtered_df.index.size - 1):
            barrier = ((sorted_examples[i] + sorted_examples[i+1])/2)
            smaller_examples_indices = filtered_df[filtered_df[f] < barrier].index
            bigger_equale_examples_indices = filtered_df[filtered_df[f] >= barrier].index

            ig = examples_entropy - ((smaller_examples_indices.size / examples_size) * self.calculateEntropy(smaller_examples_indices)
                                                    + (bigger_equale_examples_indices.size / examples_size) * self.calculateEntropy(bigger_equale_examples_indices))
            if (ig > best_ig):
                best_ig = ig
                best_barrier = barrier


        return best_ig, best_barrier

    def maxIg(self, examples_indices):
        max_ig, best_barrier = -np.inf, -1
        max_f = -1
        examples_entropy = self.calculateEntropy(examples_indices)
        for f in range(1, features_num):
            ig, barrier = self.ig(features[f], examples_indices, examples_entropy)
            if ig >= max_ig:
                max_ig, best_barrier = ig, barrier
                max_f = f
        return max_f, best_barrier

    def majorityClass(self, examples_indices):
        m = 0
        b = 0
        examples_to_iterate = np.array(train_group.filter(examples_indices, axis=0)["diagnosis"])
        for diagnosis in examples_to_iterate:
            if diagnosis == 'M':
                m += 1
            else:
                b += 1
        return 'M' if m > b else 'B'

    def isConsistentNode(self, examples_indices, majority_val):
        examples_to_iterate = np.array(train_group.filter(examples_indices, axis=0)["diagnosis"])
        for diagnosis in examples_to_iterate:
            if diagnosis != majority_val:
                return False
        return True

    def fit(self, train):
        self.decision_tree = self.tdidt(train, self.maxIg,self.majorityClass(train), False)


    def tdidt(self, examples_indices, selectFeature,default,is_early_pruning):
        if self.is_early_pruning and len(examples_indices)<self.limit:
            return Node(label=default)
        c = self.majorityClass(examples_indices)
        if self.isConsistentNode(examples_indices, c):
            return Node(label=c)
        f, barrier = selectFeature(examples_indices)
        f_col_name = features[f]
        filtered_df =train_group.filter(examples_indices, axis=0)
        l_son_examples_indices = (filtered_df[filtered_df[f_col_name]< barrier]).index
        r_son_examples_indices = (filtered_df[filtered_df[f_col_name] >= barrier]).index
        l_son = self.tdidt(l_son_examples_indices, selectFeature, c, is_early_pruning)
        r_son = self.tdidt(r_son_examples_indices, selectFeature, c, is_early_pruning)
        return Node(feature=f, barrier=barrier, l_son=l_son, r_son=r_son)

    def classifier(self,e,node):
        if node.label:
            return node.label
        if e[node.feature] < node.barrier:
            return self.classifier(e, node.l_son)
        else:
            return self.classifier(e, node.r_son)


    def predict(self, test):
        examples_num = len(test.index)
        corrects_num = 0
        examples_to_iterate = np.array(test)
        for e in examples_to_iterate:
            if self.classifier(e,self.decision_tree) == e[0]:
                corrects_num += 1
        result = corrects_num / examples_num
        return result

    def calculateEntropy(self, examples_indices):
        examples_len = len(examples_indices)
        if(examples_len == 0):
            return 0
        filtered_df = train_group.filter(examples_indices, axis=0)
        b_counter = len(filtered_df[filtered_df["diagnosis"] == 'B']) #.count()["diagnosis"]
        prob_healthy = b_counter / examples_len
        prob_sick = 1-prob_healthy
        arg1 = 0
        arg2 = 0
        if prob_sick:
            arg1 = prob_sick * np.log2(prob_sick)
        if prob_healthy:
            arg2 = prob_healthy*np.log2(prob_healthy)
        return -(arg1 + arg2)

#############end_of_class_id3####################################
    #just run it
def experiment():
    kf= KFold(n_splits=5, shuffle=True, random_state=204576946)
    splited= kf.split(train_group)
    precision_by_m = []
    for M in M_values:
        precisions_for_specific_m = []
        for train_index, test_index in splited:
            train_indices = train_index.index
            precisions_for_specific_m.append(create_experiment(train_indices, test_index, M=M))
        precision_by_m.append(np.average(precisions_for_specific_m))
    ax = plt.subplot()
    ax.plot(M_values, precision_by_m)
    ax.set(xlabel='Min examples for node decision', ylabel='Accuracy')
    plt.show()



def create_experiment(train_row_indices, test,  M, is_earlly_pruning=True):
    id3 = ID3(is_early_pruning=is_earlly_pruning, limit = M)
    id3.fit(train_row_indices)
    return id3.predict(test)

 #this is the function for ex_3_4

def ex_3_4():
    pass

def main():
    print("ID3 accuracy is:", create_experiment(train_row_indices= train_row_indices, test= test_group,M=None,is_earlly_pruning=False))



if __name__ == "__main__":
    main()
