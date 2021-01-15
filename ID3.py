import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
#import time

# from AbstractAlgorithm import AbstractAlgorithm

######### important thoughts ##############

# we dont have empty leaf problem
# we uses all features in every step to calc max_ig
#todo: change M_values
M_values = [1, 2, 3, 5, 8, 16, 30, 50, 80, 120]
#M_values = [8, 16, 30]
train_group = pd.read_csv(r"C:\Users\HP\PycharmProjects\pythonProject\hw3\train.csv")
test_group = pd.read_csv(r"C:\Users\HP\PycharmProjects\pythonProject\hw3\test.csv")
features_num = len(train_group.columns)
features = list(train_group.columns)
features.pop(0)
train_row_indices = list(range(0,len(train_group)))
train_group_dict = train_group.to_dict(orient='list')
test_row_indices = list(range(0,len(test_group)))
test_group_dict = test_group.to_dict(orient='list')

class Node:
    def __init__(self, label=None, feature=None, barrier=None, l_son=None, r_son=None):
        self.label = label
        self.feature = feature
        self.barrier = barrier
        self.l_son = l_son
        self.r_son = r_son


class ID3:
    def __init__(self, limit, is_early_pruning, predict_dict):
        self.decision_tree = None
        self.is_early_pruning= is_early_pruning
        self.limit = limit
        self.predict_dict = predict_dict

    def splitIndexs(self,f,barrier,examples_indices):
        smaller_examples_indices = []
        bigger_equale_examples_indices = []
        for idx in examples_indices:
            if train_group_dict[f][idx] < barrier:
                smaller_examples_indices.append(idx)
            else:
                bigger_equale_examples_indices.append(idx)
        return smaller_examples_indices, bigger_equale_examples_indices


    def ig(self, f, examples_indices, examples_entropy):
        values = [train_group_dict[f][idx] for idx in examples_indices]
        values.sort()
        examples_size = len(values)
        best_barrier = 0.5 * (values[0] + values[1])

        best_ig = -np.inf
        for i in range(examples_size - 1):
            barrier = ((values[i] + values[i+1])/2)

            smaller_examples_indices, bigger_equale_examples_indices = self.splitIndexs(f,barrier, examples_indices)
            ig = examples_entropy - ((len(smaller_examples_indices) / examples_size) * self.calculateEntropy(smaller_examples_indices)
                                                    + (len(bigger_equale_examples_indices) / examples_size) * self.calculateEntropy(bigger_equale_examples_indices))
            if ig > best_ig:
                best_ig = ig
                best_barrier = barrier

        return best_ig, best_barrier

    def maxIg(self, examples_indices):
        max_ig, best_barrier = -np.inf, -1
        max_f = None
        examples_entropy = self.calculateEntropy(examples_indices)
        for f in features:
            ig, barrier = self.ig(f, examples_indices, examples_entropy)
            if ig >= max_ig:
                max_ig, best_barrier = ig, barrier
                max_f = f
        return max_f, best_barrier

    def majorityClass(self, examples_indices):
        m = 0
        b = 0
        examples_to_iterate = [train_group_dict["diagnosis"][idx] for idx in examples_indices]
        for diagnosis in examples_to_iterate:
            if diagnosis == 'M':
                m += 1
            else:
                b += 1
        return 'M' if m > b else 'B'

    def isConsistentNode(self, examples_indices, majority_val):
        examples_to_iterate = [train_group_dict["diagnosis"][idx] for idx in examples_indices]
        for diagnosis in examples_to_iterate:
            if diagnosis != majority_val:
                return False
        return True

    def fit(self, train):
        self.decision_tree = self.tdidt(train, self.maxIg, self.majorityClass(train), False)

    def tdidt(self, examples_indices, selectFeature,default,is_early_pruning):
        if self.is_early_pruning and len(examples_indices) < self.limit:
            return Node(label=default)
        c = self.majorityClass(examples_indices)
        if self.isConsistentNode(examples_indices, c):
            return Node(label=c)
        f, barrier = selectFeature(examples_indices)
        l_son_examples_indices, r_son_examples_indices = self.splitIndexs(f, barrier, examples_indices)
        l_son = self.tdidt(l_son_examples_indices, selectFeature, c, is_early_pruning)
        r_son = self.tdidt(r_son_examples_indices, selectFeature, c, is_early_pruning)
        return Node(feature=f, barrier=barrier, l_son=l_son, r_son=r_son)

    def classifier(self,e_index,node):
        if node.label:
            return node.label
        if self.predict_dict[node.feature][e_index] < node.barrier:
            return self.classifier(e_index, node.l_son)
        else:
            return self.classifier(e_index, node.r_son)


    def predict(self, test_row_indices):
        examples_num = len(test_row_indices)
        corrects_num = 0
        for e_idx in test_row_indices:
            if self.classifier(e_idx, self.decision_tree) == self.predict_dict["diagnosis"][e_idx]:
                corrects_num += 1
        result = corrects_num / examples_num
        return result

    def calculateEntropy(self, examples_indices):
        examples_len = len(examples_indices)
        if(examples_len == 0):
            return 0
        diagnoses = [train_group_dict['diagnosis'][idx] for idx in examples_indices]
        b_counter = 0
        for diagnosis in diagnoses:
            if diagnosis == 'B':
                b_counter += 1
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

"""
To run uncomment the line with experiment() in mine 
"""
def experiment():
    precision_by_m = []
    for M in M_values:
        kf = KFold(n_splits=5, shuffle=True, random_state=204576946)
        splited = kf.split(train_group)
        precisions_for_specific_m = []
        for train_index, test_index in splited:
            precisions_for_specific_m.append(create_experiment(train_index, test_index, M=M))
        precision_by_m.append(np.average(precisions_for_specific_m))

    figure, ax = plt.subplots()
    ax.plot(M_values, precision_by_m, marker='o')
    ax.set(xlabel='Min examples for node decision', ylabel='Accuracy', title='Accuracy By M')
    plt.show()

def create_experiment(train_row_indices, test_row_indices,  M, is_earlly_pruning=True, predict_dict= train_group_dict):
    id3 = ID3(is_early_pruning=is_earlly_pruning, limit = M, predict_dict=predict_dict)
    id3.fit(train_row_indices)
    return id3.predict(test_row_indices)

#this is the function for ex_3_4

def ex_3_4():
    pass

def main():
    print("ID3 accuracy is:", create_experiment(train_row_indices, test_row_indices, None, False, test_group_dict))
    experiment()



if __name__ == "__main__":
    main()
