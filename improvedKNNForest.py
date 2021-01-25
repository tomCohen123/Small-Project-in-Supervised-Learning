import numpy as np
from sklearn.model_selection import KFold
import ID3 as id_three
import KNNForest as knn_forest

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

    def predict(self):
        examples_num = len(self.actual_test_indices)
        corrects_num = 0
        for e_idx in self.actual_test_indices:
            if self.forestClassifier(e_idx) == self.predict_dict["diagnosis"][e_idx]:
                corrects_num += 1
        return corrects_num / examples_num


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
            self.trees_weights.append(1 - (d / deepest_tree))

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

    def predict(self):
        examples_num = len(self.actual_test_indices)
        corrects_num = 0
        for e_idx in self.actual_test_indices:
            if self.forestClassifier(e_idx) == self.predict_dict["diagnosis"][e_idx]:
                corrects_num += 1
        return corrects_num / examples_num

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
def depth_forest_experiment():
    best_precision = -1
    best_parameters = None, None, None
    kf = KFold(n_splits=5, shuffle=True, random_state=204576946)
    for n in range(2, 10):
        for k in range(2, n + 1):
            for p in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
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

def distance_forest_experiment():
    best_precision = -1
    best_parameters = None, None, None
    kf = KFold(n_splits=5, shuffle=True, random_state=204576946)
    for n in range(2, 10):
        for k in range(2, n + 1):
            for p in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
                precisions_for_specific_third = []
                for train_index, test_index in kf.split(id_three.train_group):
                    forest = DistanceWeightedKnnForest(n, k, p, predict_dict=id_three.train_group_dict,
                                                    actual_train_indices=train_index,
                                                    actual_test_indices=test_index, consistent_node_threshold=0.99)
                    forest.fit()
                    precisions_for_specific_third.append(forest.predict())
                if np.average(precisions_for_specific_third) > best_precision:
                    best_parameters = n, k, p
    return best_parameters


def main():
    """the commented lines before belongs to tests i ran to find best parameters"""
    # print("depth forest experience result:", depth_forest_experiment())
    # print("distance forest experience result:", distance_forest_experiment())

    """training 1st forest- depth weighted forest"""
    depth_forest = DepthWeightedKnnForest(9, 9, 0.3, predict_dict=id_three.test_group_dict,
                       actual_train_indices=id_three.train_row_indices,
                       actual_test_indices=id_three.test_row_indices, consistent_node_threshold=0.99)
    depth_forest.improved_fit()
    """training 2nd forest- distance weighted forest"""
    distance_forest = DistanceWeightedKnnForest(9, 9, 0.3, predict_dict=id_three.test_group_dict,
                                          actual_train_indices=id_three.train_row_indices,
                                          actual_test_indices=id_three.test_row_indices, consistent_node_threshold=0.99)
    distance_forest.fit()
    """training 3rd forest- non weighted forest"""
    normal_forest = NormalKnnForest(9, 9, 0.3, predict_dict=id_three.test_group_dict,
                                                actual_train_indices=id_three.train_row_indices,
                                                actual_test_indices=id_three.test_row_indices,
                                                consistent_node_threshold=0.99)
    normal_forest.fit()
    """Now our ImprovedKnnForest will classify each example as the majority of classification of all trees"""
    examples_num = len(id_three.test_row_indices)
    corrects_num = 0
    for e_idx in id_three.test_row_indices:
        classifications=[]
        classifications.append((depth_forest.forestClassifier(e_idx),1))
        classifications.append((distance_forest.forestClassifier(e_idx), 1))
        classifications.append((normal_forest.forestClassifier(e_idx), 1))
        if depth_forest.calculateMajorityClass(classifications) == id_three.test_group_dict["diagnosis"][e_idx]:
            corrects_num += 1
    print(corrects_num / examples_num)

if __name__ == "__main__":
    main()