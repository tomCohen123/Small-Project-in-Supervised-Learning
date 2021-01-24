import time
import numpy as np
from sklearn.model_selection import KFold
import ID3 as id_three
import KNNForest as knn_forest
#ideas:
# use late pruning on each tree if there is enough time

#look for best group v for late pruning with barrier and random

#give different weight to features some how

#find best train subgroup from train with random size and places use k fold

#best values for forest:
# N=9
# K=5
# P=0.3

#todo: does tree weights really work?
#todo: delte import time and time from main

"""my improved knn forest is implemented as a main function that returns the majority classification of the
 classifications determined by the next 3 Forests:
 1.normal forest- same as KNNForest that we had to implement in previous question
 (except isConsistentNode func as explained in report)
 2.distance weighted forest - give to each classification of a tree from the Knn chosen trees a weight by its distance
 to the given test example
 3.depth weighted forest - give to each classification of a tree from the Knn chosen trees a weight by its depth
 
 In this file you will find first a 3 classes which implement the 3 Forests above and finally the main function 
 described above.
 """


#normalization of paramerter
#special k fold with average of n average of k average of p to choose parameters on combined data, explain in dry the new k fold style

class NormalKnnForest(knn_forest.KNNForest):
    def __init__(self, n, k, p, predict_dict, actual_train_indices, actual_test_indices, consistent_node_threshold):
        knn_forest.KNNForest.__init__(self, n, k, p, predict_dict, actual_train_indices, actual_test_indices)
        self.consistent_node_threshold = consistent_node_threshold

    def isConsistentNode(self, examples_indices, majority_val):
        examples_to_iterate = [id_three.train_group_dict["diagnosis"][idx] for idx in examples_indices]
        majority_counter = 0
        for diagnosis in examples_to_iterate:
            if diagnosis == majority_val:
                majority_counter += 1
        return True if majority_counter / len(examples_indices) >= self.consistent_node_threshold else False

    def forestClassifier(self,e_idx):
        knn_indices = self.chooseKnnTreesIndices(e_idx)
        classify_results = []
        for tree_idx in knn_indices:
            tree = self.trees_list[tree_idx]
            classify_results.append(tree.classifier(e_idx, tree.decision_tree))
        return self.calculateMajorityClass(classify_results)

class DistanceWeightedKnnForest(knn_forest.KNNForest):
    def __init__(self, n, k, p, predict_dict, actual_train_indices, actual_test_indices, consistent_node_threshold):
        knn_forest.KNNForest.__init__(self, n, k, p, predict_dict, actual_train_indices, actual_test_indices)
        self.consistent_node_threshold = consistent_node_threshold

    def isConsistentNode(self, examples_indices, majority_val):
        examples_to_iterate = [id_three.train_group_dict["diagnosis"][idx] for idx in examples_indices]
        majority_counter = 0
        for diagnosis in examples_to_iterate:
            if diagnosis == majority_val:
                majority_counter += 1
        return True if majority_counter / len(examples_indices) >= self.consistent_node_threshold else False

    def calculateMajorityClass(self, classify_results):
        m = 0
        b = 0
        for diagnosis in classify_results:
            if diagnosis[0] == 'M':
                m += diagnosis[1]
            else:
                b += diagnosis[1]
        return 'M' if m > b else 'B'

    def forestClassifier(self,e_idx):
        knn_indices = self.chooseKnnTreesIndices(e_idx)
        classify_results = []
        weighter = 0
        for tree_idx in knn_indices:
            tree = self.trees_list[tree_idx]
            classify_results.append((tree.classifier(e_idx, tree.decision_tree), (self.K - weighter) / self.K))
            weighter += 1
        return self.calculateMajorityClass(classify_results)

    # def predict(self):
    #     examples_num = len(self.actual_test_indices)
    #     corrects_num = 0
    #     for e_idx in self.actual_test_indices:
    #         knn_indices = self.chooseKnnTreesIndices(e_idx)
    #         classify_results = []
    #         weighter = 0
    #         for tree_idx in knn_indices:
    #             tree = self.trees_list[tree_idx]
    #             classify_results.append((tree.classifier(e_idx, tree.decision_tree), (self.K-weighter)/self.K))
    #             weighter += 1
    #
    #         if self.calculateMajorityClass(classify_results) == self.predict_dict["diagnosis"][e_idx]:
    #             corrects_num += 1
    #     return corrects_num / examples_num


class DepthWeightedKnnForest(knn_forest.KNNForest):
    def __init__(self, n, k, p, predict_dict, actual_train_indices, actual_test_indices, consistent_node_threshold):
        knn_forest.KNNForest.__init__(self, n, k, p, predict_dict, actual_train_indices, actual_test_indices)
        self.consistent_node_threshold = consistent_node_threshold
        self.trees_weights = []

    def isConsistentNode(self, examples_indices, majority_val):
        examples_to_iterate = [id_three.train_group_dict["diagnosis"][idx] for idx in examples_indices]
        majority_counter = 0
        for diagnosis in examples_to_iterate:
            if diagnosis == majority_val:
                majority_counter += 1
        return True if majority_counter / len(examples_indices) >= self.consistent_node_threshold else False

    def update_trees_weight(self):
        trees_depths = []
        for tree in self.trees_list:
            trees_depths.append(depth(tree.decision_tree))
        deepest_tree = max(trees_depths)
        for d in trees_depths:
            self.trees_weights.append(1 - d / deepest_tree)

    def improved_fit(self):
        self.fit()
        self.update_trees_weight()

    def calculateMajorityClass(self, classify_results):
        m = 0
        b = 0
        for diagnosis in classify_results:
            if diagnosis[0] == 'M':
                m += diagnosis[1]
            else:
                b += diagnosis[1]
        return 'M' if m > b else 'B'

    # def predict(self):
    #     examples_num = len(self.actual_test_indices)
    #     corrects_num = 0
    #     for e_idx in self.actual_test_indices:
    #         knn_indices = self.chooseKnnTreesIndices(e_idx)
    #         classify_results = []
    #         for tree_idx in knn_indices:
    #             tree = self.trees_list[tree_idx]
    #             classify_results.append((tree.classifier(e_idx, tree.decision_tree), self.trees_weights[tree_idx]))
    #         if self.calculateMajorityClass(classify_results) == self.predict_dict["diagnosis"][e_idx]:
    #             corrects_num += 1
    #     return corrects_num / examples_num

    def forestClassifier(self, e_idx):
        knn_indices = self.chooseKnnTreesIndices(e_idx)
        classify_results = []
        weighter = 0
        for tree_idx in knn_indices:
            tree = self.trees_list[tree_idx]
            classify_results.append((tree.classifier(e_idx, tree.decision_tree), self.trees_weights[tree_idx]))
            weighter += 1
        return self.calculateMajorityClass(classify_results)

def depth(node):
    if node is None:
        return 0
    l_depth = depth(node.l_son)
    r_depth = depth(node.r_son)
    return 1+max(l_depth, r_depth)


"""
main run time is 20 sec
the comment line in main is the experiment i use to calculate best parameters
"""
def experiments():
    best_precision = -1
    best_parameters = None, None, None
    kf = KFold(n_splits=5, shuffle=True, random_state=204576946)
    for n in range(2, 20):
        for k in range(2, n + 1):
            for p in [0.3,0.33,0.35,0.37,0.4,0.43,0.45,0.47,0.5,0.53, 0.55,0.57, 0.6, 0.63,0.65,0.67,0.7]:
                precisions_for_specific_third = []
                for train_index, test_index in kf.split(id_three.train_group):
                    forest = DepthWeightedKnnForest(n, k, p, predict_dict=id_three.train_group_dict,
                                       actual_train_indices=train_index,
                                       actual_test_indices=test_index,consistent_node_threshold=0.99)
                    forest.improved_fit()
                    precisions_for_specific_third.append(forest.predict())
                if np.average(precisions_for_specific_third) > best_precision:
                    best_parameters = n, k, p
    return best_parameters

def main():
    start = time.time()
    #print(experiments())

    #todo: verifay that below is best parameters
    #todo: uncomment below
    """training 1st forest- depth weighted forest"""
    depth_forest = DepthWeightedKnnForest(8, 5, 0.3, predict_dict=id_three.test_group_dict,
                       actual_train_indices=id_three.train_row_indices,
                       actual_test_indices=id_three.test_row_indices, consistent_node_threshold=0.99)
    depth_forest.improved_fit()
    """training 2nd forest- distance weighted forest"""
    distance_forest = DistanceWeightedKnnForest(8, 5, 0.3, predict_dict=id_three.test_group_dict,
                                          actual_train_indices=id_three.train_row_indices,
                                          actual_test_indices=id_three.test_row_indices, consistent_node_threshold=0.99)
    distance_forest.fit()
    """training 3rd forest- non weighted forest"""
    normal_forest = NormalKnnForest(8, 5, 0.3, predict_dict=id_three.test_group_dict,
                                                actual_train_indices=id_three.train_row_indices,
                                                actual_test_indices=id_three.test_row_indices,
                                                consistent_node_threshold=0.99)
    normal_forest.fit()
    """Now our ImprovedKnnForest will classify each example as the majority of classification of all trees"""
    examples_num = len(id_three.test_row_indices)
    corrects_num = 0
    for e_idx in id_three.test_row_indices:
        classifications=[]
        classifications.append(depth_forest.forestClassifier(e_idx))
        classifications.append(distance_forest.forestClassifier(e_idx))
        classifications.append(normal_forest.forestClassifier(e_idx))
        if normal_forest.calculateMajorityClass(classifications) == id_three.test_group_dict["diagnosis"][e_idx]:
            corrects_num += 1
    print(time.time() - start)
    print(corrects_num / examples_num)

if __name__ == "__main__":
    main()