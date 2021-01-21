import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("train.csv")
train_group_dict = df.to_dict(orient='list')
# features = df.columns
# features.pop(0)

print(range(0.3,0.7,0.1))

# l = df.loc[0]
# vec1=[]
# vec2=[]
# for i in range(9):
#    vec1.append(1)
#    vec2.append(2)
# print(np.linalg.norm(np.array(vec1) - np.array(vec2)))

# examples_indices = [1,8,3]
# a=df.filter(examples_indices, axis=0)#["diagnosis"]
# index_list = a.index
# print(train_group_dict['diagnosis'][0])
# print(train_group_dict['diagnosis'][1])
# features = df.columns
#
#
# print(features)

# list = [14,2,4]
# list.sort()
# print(0.5*(list[0]+list[1]))
# print(list)

#df = np.genfromtxt(r"C:\Users\HP\PycharmProjects\pythonProject\hw3\train.csv", delimiter=',',)
#print(df[0])
#sicks = df[df['diagnosis'] == 'M']

# #for x in range(1,31) :
#     print (x)

# f_col_name = df.columns[0]
#sicks = df[df["diagnosis"] == 'M']
#
#print(type(df))
# for index in df.index:
#     print(df.loc[index]['diagnosis'])

#sorted = df.sort_values('radius_mean')
#print(df[df["diagnosis"] == 'M'].count()["diagnosis"])

# one_columon = df["radius_mean"].sort_values()
# print(one_columon)
# print(one_columon[0])
#


#print(df)


#print(sicks)
# print(df.loc[0]['diagnosis'])



""" late pruning function that work with costSensitiveId3 


    def latePruning(self, node, v):
        if node.l_son is None and node.r_son is None:
            return node

        smaller_v, bigger_equale_v = id_three.splitIndexs(f=node.feature, barrier=node.barrier, examples_indices=v)
        node.l_son = self.latePruning(node.l_son, smaller_v)
        node.r_son = self.latePruning(node.r_son, bigger_equale_v)

        err_prune = 0
        err_no_prune = 0
        for idx in v:
            v_idx_label = id_three.train_group_dict['diagnosis'][idx]
            err_prune += evaluate(v_idx_label, node.label)
            err_no_prune += evaluate(v_idx_label, self.classifier(idx, node))

        if err_prune < err_no_prune:
            node.f = None
            node.l_son = None
            node.r_son = None

        return node

#define below out of class cost sensitive!

def evaluate(v_idx_label, tree_label):
    if v_idx_label == tree_label:
        return 0
    return 9 if v_idx_label == 'M' else 1

#tets that dont work for late pruning !



v must be a part of train test. Thus we need to split train into 2 groups: actual training group and validation group.
I started from taking first 50 indices of train for validation because it seemed to me
a logical number, and the rest to actual training.
Then each iteration i took the 50 next indices from the actual train group and add them to the validation group,  
until 50 indexes left in the train group. 

# def chooseTrainExperiment():
#     # experiment_id3 = CostSensitiveID3(is_early_pruning=False, limit=None, predict_dict=id_three.train_group_dict)
#     # threshold = 1
#     # ###loss is between 0 to 1 thus if i intialized it to that value to be sure best_loss will be updated in first iteration.
#     # best_loss = 2
#     # best_threshold = 1
#     # train_group_len = len(id_three.train_row_indices)
#     #
#     # while train_group_len-threshold >= 30:
#     #
#     #
#     #     experiment_id3.fit(id_three.train_row_indices[threshold:])
#     #     experiment_id3.decision_tree = experiment_id3.latePruning(experiment_id3.decision_tree, id_three.train_row_indices[:50])
#     #     cur_loss = experiment_id3.predictLoss(id_three.train_row_indices[50:100])
#     #     if cur_loss < best_loss:
#     #         best_loss = cur_loss
#     #         best_threshold = threshold
#     #     threshold += 50
#     # print(best_threshold)
#     # print(best_loss)
#
#     always first 50 indices are using for validation group, and the 50 after them to check loss"
#     experiment_id3 = CostSensitiveID3(is_early_pruning=False, limit=None, predict_dict=id_three.train_group_dict)
#     experiment with train with size 100
#     best_loss = 2
#     best_vector = None
#     for i in range(50):
#         id3_inst_train_indices = random.sample(id_three.train_row_indices[100:], 100)
#         experiment_id3.fit(id3_inst_train_indices)
#         #experiment_id3.decision_tree = experiment_id3.latePruning(experiment_id3.decision_tree, id_three.train_row_indices[:50])
#         cur_loss = experiment_id3.predictLoss(id_three.train_row_indices[50:100])
#         if cur_loss < best_loss:
#             best_loss = cur_loss
#             best_vector = id3_inst_train_indices
#     return best_vector

"""
