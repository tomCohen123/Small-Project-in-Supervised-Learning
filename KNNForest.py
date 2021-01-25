import ID3 as id_three
import random
import numpy as np
from sklearn.model_selection import KFold

"""Each Tree in the forest is Ide with no early pruning because i found it is better"""

class ForestID3(id_three.ID3):
    def __init__(self,  is_early_pruning, limit, predict_dict):
        id_three.ID3.__init__(self,limit=limit, is_early_pruning= is_early_pruning, predict_dict= predict_dict)
        self.centroid = None

    def setCentroid(self, train_indices):
        centroid = []
        for f in id_three.features:
            f_values = [id_three.train_group_dict[f][idx] for idx in train_indices]
            centroid.append(np.average(f_values))
        self.centroid = centroid



class KNNForest:
    def __init__(self, n, k, p, predict_dict, actual_train_indices, actual_test_indices):
        self.trees_list = []
        self.N = n
        self.K = k
        self.P = p
        self.predict_dict = predict_dict
        self.actual_train_indices = actual_train_indices
        self.actual_test_indices = actual_test_indices

    def calculateMajorityClass(self,classify_results):
        m = 0
        b = 0
        for diagnosis in classify_results:
            if diagnosis == 'M':
                m += 1
            else:
                b += 1
        return 'M' if m > b else 'B'

    def fit(self):
        for i in range(self.N):
            self.trees_list.append(ForestID3(is_early_pruning=False, limit=None, predict_dict=self.predict_dict))
        for id3_inst in self.trees_list:
            id3_inst_train_indices = random.sample(list(self.actual_train_indices), int(self.P * len(self.actual_train_indices)))
            id3_inst.fit(id3_inst_train_indices)
            id3_inst.setCentroid(id3_inst_train_indices)

    def chooseKnnTreesIndices(self, e_idx):
        index_and_dist = []
        e_idx_values = [self.predict_dict[f][e_idx] for f in id_three.features]
        for idx in range(len(self.trees_list)):
            index_and_dist.append(
                (idx, np.linalg.norm(np.array(self.trees_list[idx].centroid) - np.array(e_idx_values))))
        index_and_dist.sort(key=lambda x: x[1])
        indices = []
        for i in range(self.K):
            indices.append(index_and_dist[i][0])
        return indices

    def predict(self):
        examples_num = len(self.actual_test_indices)
        corrects_num = 0

        for e_idx in self.actual_test_indices:
            knn_indices = self.chooseKnnTreesIndices(e_idx)
            classify_results = []
            for tree_idx in knn_indices:
                tree = self.trees_list[tree_idx]
                classify_results.append(tree.classifier(e_idx, tree.decision_tree))
            if self.calculateMajorityClass(classify_results) == self.predict_dict["diagnosis"][e_idx]:
                corrects_num += 1
        return corrects_num / examples_num




"""Below is my experiment to set the best parameters for the algorithm, i am using k-fold to estimate the perscion
    of every parameters groups and returns the one which maximaize the precision"""

def experiments():
    best_precision = -1
    best_parameters = None, None, None
    kf = KFold(n_splits=5, shuffle=True, random_state=204576946)
    for n in range(2, 10):
        for k in range(2, n + 1):
            for p in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
                precisions_for_specific_third = []
                for train_index, test_index in kf.split(id_three.train_group):
                    forest = KNNForest(n, k, p, predict_dict=id_three.train_group_dict,
                                                       actual_train_indices=train_index,
                                                       actual_test_indices=test_index)
                    forest.fit()
                    precisions_for_specific_third.append(forest.predict())
                if np.average(precisions_for_specific_third) > best_precision:
                    best_parameters = n, k, p
    return best_parameters


"""
the commented line in main is the experiment i use to calculate best parameters
"""
def main():
    #print(experiments())
    forest = KNNForest(9, 9, 0.7, predict_dict=id_three.test_group_dict,
                       actual_train_indices=id_three.train_row_indices,
                       actual_test_indices=id_three.test_row_indices)
    forest.fit()
    print(forest.predict())

if __name__ == "__main__":
    main()
