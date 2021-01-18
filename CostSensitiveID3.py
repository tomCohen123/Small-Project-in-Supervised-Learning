import ID3 as id_three

# todo: what experiments?
# todo: train data preprocessing?
# todo: change v group to be part of train
# todo: use a change of entropy?

#id3 = id_three.ID3(is_early_pruning=False, limit=None, predict_dict=id_three.test_group_dict)

class CostSensitiveID3(id_three.ID3):
    def __init__(self,  is_early_pruning, limit, predict_dict, ):
        id_three.ID3.__init__(self,limit=limit, is_early_pruning= is_early_pruning, predict_dict= predict_dict)

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

def evaluate(v_idx_label, tree_label):
    if v_idx_label == tree_label:
        return 0
    return 9 if v_idx_label == 'M' else 1




"""
v must be a part of train test. Thus we need to split train into 2 groups: actual training group and validation group.
I started from taking first 50 indices of train for validation because it seemed to me
a logical number, and the rest to actual training.
Then each iteration i took the 50 next indices from the actual train group and add them to the validation group,  
until 50 indexes left in the train group. 
"""
def chooseVExperiment():
    experiment_id3 = CostSensitiveID3(is_early_pruning=False, limit=None, predict_dict=id_three.train_group_dict)
    threshold = 1
    ###loss is between 0 to 1 thus if i intialized it to that value to be sure best_loss will be updated in first iteration.
    best_loss = 2
    best_threshold = 1
    train_group_len = len(id_three.train_row_indices)

    while train_group_len-threshold >= 30:
        experiment_id3.fit(id_three.train_row_indices[threshold:])
        experiment_id3.decision_tree = experiment_id3.latePruning(experiment_id3.decision_tree, id_three.train_row_indices[:50])
        cur_loss = experiment_id3.predictLoss(id_three.train_row_indices[50:100])
        if cur_loss < best_loss:
            best_loss = cur_loss
            best_threshold = threshold
        threshold += 50
    print(best_threshold)
    print(best_loss)



def main():
    # because my optimal m was 1, it means no prune, i use the algorithem with no pruning
#todo: uncomment below
    id3 = CostSensitiveID3(is_early_pruning=False, limit=None, predict_dict=id_three.train_group_dict)
    id3.fit(id_three.train_row_indices[50:])
    id3.decision_tree = id3.latePruning(id3.decision_tree, id_three.train_row_indices[:50])
    id3.predict_dict=id_three.test_group_dict
    print(id3.predictLoss(id_three.test_row_indices))
# todo: uncomment above

# todo: comment below
    #chooseVExperiment()
# todo: comment above
if __name__ == "__main__":
    main()
