import pandas as pd
import numpy as np
#import time


# from AbstractAlgorithm import AbstractAlgorithm

######### important thoughts ##############

# we dont have empty leaf problem
# we uses all features in every step to calc max_ig

class Node:
    def __init__(self, label=None, feature=None, barrier=None, l_son=None, r_son=None):
        self.label = label
        self.feature = feature
        self.barrier = barrier
        self.l_son = l_son
        self.r_son = r_son


class ID3:
    def __init__(self):
        self.train_group = pd.read_csv(r"C:\Users\HP\PycharmProjects\pythonProject\hw3\train.csv")
        self.test_group = pd.read_csv(r"C:\Users\HP\PycharmProjects\pythonProject\hw3\test.csv")
        self.features_num = len(self.train_group.columns)
        self.decision_tree = None

    def ig(self, f, examples, examples_entropy):
        sorted_examples = (examples[f].sort_values()).reset_index(drop=True)
        examples_size = len(sorted_examples)

        best_barrier = 0.5 * (sorted_examples[0] + sorted_examples[1])

        best_ig = -np.inf
        for i in range(examples.index.size - 1):
            barrier = ((sorted_examples[i] + sorted_examples[i+1])/2)
            smaller_examples = examples[examples[f] < barrier]
            bigger_equale_examples = examples[examples[f] >= barrier]

            ig = examples_entropy - ((len(smaller_examples) / examples_size) * self.calculateEntropy(smaller_examples)
                                                    + (len(bigger_equale_examples) / examples_size) * self.calculateEntropy(bigger_equale_examples))
            if (ig > best_ig):
                best_ig = ig
                best_barrier = barrier


        return best_ig, best_barrier

    def maxIg(self, examples):
        max_ig, best_barrier = -np.inf, -1
        max_f = -1
        examples_entropy = self.calculateEntropy(examples)
        for f in range(1, self.features_num):
            ig, barrier = self.ig(examples.columns[f], examples, examples_entropy)
            if ig >= max_ig:
                max_ig, best_barrier = ig, barrier
                max_f = f
        return max_f, best_barrier

    def majorityClass(self, examples):
        m = 0
        b = 0
        examples_to_iterate = np.array(examples["diagnosis"])
        for diagnosis in examples_to_iterate:
            if diagnosis == 'M':
                m += 1
            else:
                b += 1
        return 'M' if m > b else 'B'

    def isConsistentNode(self, examples, majority_val):
        examples_to_iterate = np.array(examples["diagnosis"])
        for diagnosis in examples_to_iterate:
            if diagnosis != majority_val:
                return False
        return True

    def fit(self):
        self.decision_tree = self.tdidt(self.train_group, self.maxIg)


    def tdidt(self, examples, selectFeature):

        c = self.majorityClass(examples)
        if self.isConsistentNode(examples, c):
            return Node(label=c)
        f, barrier = selectFeature(examples)
        f_col_name = examples.columns[f]
        l_son_examples = examples[examples[f_col_name] < barrier]
        r_son_exaples = examples[examples[f_col_name] >= barrier]
        l_son = self.tdidt(l_son_examples, selectFeature)
        r_son = self.tdidt(r_son_exaples, selectFeature)
        return Node(feature=f, barrier=barrier, l_son=l_son, r_son=r_son)

    def classifier(self,e,node):
        if node.label:
            return node.label
        if e[node.feature] < node.barrier:
            return self.classifier(e, node.l_son)
        else:
            return self.classifier(e, node.r_son)


    def predict(self):
        examples_num = len(self.test_group.index)
        corrects_num = 0
        examples_to_iterate = np.array(self.test_group)
        for e in examples_to_iterate:
            if self.classifier(e,self.decision_tree) == e[0]:
                corrects_num += 1
        print("ID3 accuracy is:", corrects_num / examples_num)

    def calculateEntropy(self, examples):
        examples_len = len(examples)
        if(examples_len == 0):
            return 0
        b_counter = len(examples[examples["diagnosis"] == 'B']) #.count()["diagnosis"]
        prob_healthy = b_counter / examples_len
        prob_sick = 1-prob_healthy
        arg1 = 0
        arg2 = 0
        if prob_sick:
            arg1 = prob_sick * np.log2(prob_sick)
        if prob_healthy:
            arg2 = prob_healthy*np.log2(prob_healthy)
        return -(arg1 + arg2)



def main():
    id3 = ID3()
    id3.fit()
    id3.predict()


if __name__ == "__main__":
    main()
