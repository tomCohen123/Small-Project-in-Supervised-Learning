import ID3 as id_three
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

"""data preprocessing is allowed"""

class CostSensitiveID3(id_three.ID3):
    def __init__(self,  is_early_pruning, limit, predict_dict,consistent_node_threshold):
        id_three.ID3.__init__(self,limit=limit, is_early_pruning= is_early_pruning, predict_dict= predict_dict)
        self.consistent_node_threshold = consistent_node_threshold

    def isConsistentNode(self, examples_indices, majority_val):
        examples_to_iterate = [id_three.train_group_dict["diagnosis"][idx] for idx in examples_indices]
        m_counter = 0
        for diagnosis in examples_to_iterate:
            if diagnosis == 'M':
                m_counter += 1
        if m_counter/len(examples_indices) >= self.consistent_node_threshold:
            return True
        else:
            return True if m_counter == 0 else False

    def calculateEntropy(self, examples_indices):
         examples_len = len(examples_indices)
         if(examples_len == 0):
             return 0
         diagnoses = [id_three.train_group_dict['diagnosis'][idx] for idx in examples_indices]
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

         return 0 if prob_sick >= self.consistent_node_threshold else -(arg1 + arg2)


def filterTrainDataByEuclideanDist(euclidean_dist_threshold, filter_start_idx):
    delete_from_database = []
    for fr_idx in id_three.train_row_indices[filter_start_idx:]:
        first_row_f_values = np.array([id_three.train_group_dict[f][fr_idx] for f in id_three.features])
        first_row_diagnosis = id_three.train_group_dict["diagnosis"][fr_idx]
        close_and_different_label = []
        for r_idx in id_three.train_row_indices[fr_idx + 1:]:
            f_values = np.array([id_three.train_group_dict[f][r_idx] for f in id_three.features])
            f_diagnosis = id_three.train_group_dict["diagnosis"][r_idx]
            if np.linalg.norm(
                    first_row_f_values - f_values) < euclidean_dist_threshold and first_row_diagnosis != f_diagnosis:
                close_and_different_label.append(r_idx)
        if len(close_and_different_label):
            delete_from_database += close_and_different_label
            delete_from_database.append(fr_idx)

    # i new that there are rows that are close with different diagnosis so i appended first row
    return set(delete_from_database)

"""
in the experiments i use k fold valdiation on train group  with k= 5 to choose:
1.best threshold for determining if node is consistent (i use it to determine if a node is a leaf and for entropy calculation
2.best minimum Euclidean distance for removing 2 examples that are closer than that distance and have different labels,
I do both searches to decrease over fitting to train data set.      
"""
def experiments():
    """1. - best threshold experiment"""
    best_loss = 2
    best_threshold = None
    kf = KFold(n_splits=5, shuffle=True, random_state=204576946)
    average_lost_list = []
    for threshold in [0.95, 0.96, 0.97, 0.98, 0.99]:
        id3 = CostSensitiveID3(False, None, id_three.train_group_dict, threshold)
        loss_list = []
        for train_index, test_index in kf.split(id_three.train_group):
            id3.fit(train_index)
            loss_list.append(id3.predictLoss(test_index))
        everaged_loss = np.average(loss_list)
        average_lost_list.append(everaged_loss)
        if everaged_loss < best_loss:
            best_loss = everaged_loss
            best_threshold = threshold

    """drawing graph for threshold experiment"""
    figure, ax = plt.subplots()
    ax.plot([0.95, 0.96, 0.97, 0.98, 0.99], average_lost_list, marker='o')
    ax.set(xlabel='threshold', ylabel='loss', title='loss By threshold')
    plt.show()

    """2. - best euclidean dist experiment"""
    best_loss = 2
    best_euclidean_dist = None
    id3 = CostSensitiveID3(False, None, id_three.train_group_dict, 1)
    validation_group = id_three.train_row_indices[:50]
    dist_loss_list = []
    for euclidean_dist in [100, 125, 150, 175, 200]:
        # i gave a threshold of 1 to run algorithm with no change in entropy calculation and is consistent node determination
        id3.fit(list(set(id_three.train_row_indices[50:]) - filterTrainDataByEuclideanDist(euclidean_dist,50)))
        cur_loss = id3.predictLoss(validation_group)
        dist_loss_list.append(cur_loss)
        if cur_loss < best_loss:
            best_loss = cur_loss
            best_euclidean_dist = euclidean_dist

    """drawing graph for threshold experiment"""
    figure, ax = plt.subplots()
    ax.plot([100, 125, 150, 175, 200], dist_loss_list, marker='o')
    ax.set(xlabel='euclidean dist threshold', ylabel='loss', title='loss By euclidean dist threshold')
    plt.show()

    return best_threshold, best_euclidean_dist


def main():
    # because my optimal m was 1, it means no prune, i use the algorithm with no pruning
    id3 = CostSensitiveID3(is_early_pruning=False, limit=None, predict_dict=id_three.train_group_dict,
                           consistent_node_threshold=0.99)
    id3.fit(list(set(id_three.train_row_indices)-filterTrainDataByEuclideanDist(200, 0)))
    id3.predict_dict=id_three.test_group_dict
    print(id3.predictLoss(id_three.test_row_indices))
    print(experiments())


if __name__ == "__main__":
    main()
