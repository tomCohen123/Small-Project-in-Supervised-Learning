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

"""data pre processing is prohibited"""
"""loss is not interesting"""

#normalization of paramerter
#special k fold with average of n average of k average of p to choose parameters on combined data, explain in dry the new k fold style


import KNNForest as knn_forest
import ID3 as id_three




class ImprovedKnnForest(knn_forest.KNNForest):
    def __init__(self, n, k, p, predict_dict, actual_train_indices, actual_test_indices, consistent_node_threshold):
        knn_forest.KNNForest.__init__(self, n, k, p, predict_dict, actual_train_indices, actual_test_indices)
        self.consistent_node_threshold = consistent_node_threshold
        self.trees_depths = []
        self.trees_weights = []
        self.deepest_tree = None



    def isConsistentNode(self, examples_indices, majority_val):
        examples_to_iterate = [id_three.train_group_dict["diagnosis"][idx] for idx in examples_indices]
        majority_counter = 0
        for diagnosis in examples_to_iterate:
            if diagnosis == majority_val:
                majority_counter += 1
        return True if majority_counter / len(examples_indices) >= self.consistent_node_threshold else False

    def update_trees_length(self):
        for tree in self.trees_list:
            self.trees_depths.append(depth(tree.decision_tree))
        self.deepest_tree = max(self.trees_depths)

    def update_trees_weight(self):
        for depth in self.trees_depths:
            self.trees_weights.append(1-depth/self.deepest_tree)

    def improved_fit(self):
        self.fit()
        self.update_trees_length()
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
            knn_indices = self.chooseKnnTreesIndices(e_idx)
            classify_results = []
            for tree_idx in knn_indices:
                tree = self.trees_list[tree_idx]
                classify_results.append((tree.classifier(e_idx, tree.decision_tree) ,self.trees_weights[tree_idx]))
            if self.calculateMajorityClass(classify_results) == self.predict_dict["diagnosis"][e_idx]:
                corrects_num += 1
        return corrects_num / examples_num



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
    pass

def main():
    #print(experiments())
    #todo: verifay that below is best parameters
    forest = ImprovedKnnForest(10, 7, 0.32, predict_dict=id_three.test_group_dict,
                       actual_train_indices=id_three.train_row_indices,
                       actual_test_indices=id_three.test_row_indices, consistent_node_threshold=0.99)
    forest.improved_fit()
    print(forest.predict())

if __name__ == "__main__":
    main()