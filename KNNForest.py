import ID3 as id_three
import random
import numpy as np
import time

N=9
K=5
P=0.3
train_size = len(id_three.train_row_indices)


# todo: what experiments?
# todo: train data preprocessing?
# todo: delete time import and time in main
"""
# I chose to use Id3 tree with no early pruning because i 
"""

id3_global_list=[]

class ForestID3(id_three.ID3):
    def __init__(self,  is_early_pruning, limit, predict_dict, ):
        id_three.ID3.__init__(self,limit=limit, is_early_pruning= is_early_pruning, predict_dict= predict_dict)
        self.centroid = None

    def setCentroid(self, train_indices):
        centroid = []
        for f in id_three.features:
            f_values = [id_three.train_group_dict[f][idx] for idx in train_indices]
            centroid.append(np.average(f_values))
        self.centroid = centroid

def calaulateMajorityClass(classify_results):
    m = 0
    b = 0
    for diagnosis in classify_results:
        if diagnosis == 'M':
            m += 1
        else:
            b += 1
    return 'M' if m > b else 'B'

def fit():
    for i in range(N):
        id3_global_list.append(ForestID3(is_early_pruning=False, limit=None, predict_dict=id_three.test_group_dict))
    for id3_inst in id3_global_list:
        id3_inst_train_indices = random.sample(id_three.train_row_indices, int(P*train_size))
        id3_inst.fit(id3_inst_train_indices)
        id3_inst.setCentroid(id3_inst_train_indices)

def chooseKnnTreesIndices(e_idx):
    index_and_dist = []
    e_idx_values = [id_three.test_group_dict[f][e_idx] for f in id_three.features]
    for idx in range(len(id3_global_list)):
        index_and_dist.append((idx, np.linalg.norm(np.array(id3_global_list[idx].centroid) - np.array(e_idx_values))))
    index_and_dist.sort(key = lambda x: x[1])
    indices = []
    for i in range(K):
        indices.append(index_and_dist[i][0])
    return indices






def predict():
    examples_num = len(id_three.test_row_indices)
    corrects_num = 0

    for e_idx in id_three.test_row_indices:
        knn_indices = chooseKnnTreesIndices(e_idx)
        classify_results=[]
        for tree_idx in knn_indices:
            tree = id3_global_list[tree_idx]
            classify_results.append(tree.classifier(e_idx,tree.decision_tree))
        if calaulateMajorityClass(classify_results) == id_three.test_group_dict["diagnosis"][e_idx]:
            corrects_num+=1
    return corrects_num / examples_num


# i used this function to determine best N and K
# here i can combine all data

def experiments():
    N = 1
    K=1
    P=1
    fit()
    print(predict())


def main():
    #start = time.time()
    fit()
    print(predict())
    experiments()
    #print(time.time()-start)

if __name__ == "__main__":
    main()
