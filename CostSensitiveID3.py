import ID3 as id_three

# todo: what experiments?
# todo: train data preprocessing?
# todo: change v group to be part of train

id3 = id_three.ID3(is_early_pruning=False, limit=None, predict_dict=id_three.test_group_dict)


def evaluate(v_idx_label, tree_label):
    if v_idx_label == tree_label:
        return 0
    return 9 if v_idx_label == 'M' else 1


def latePruning(node, v):
    if node.l_son==None and node.r_son==None:
        return node

    smaller_v, bigger_equale_v = id_three.splitIndexs(f=node.feature, barrier=node.barrier, examples_indices=v)
    node.l_son = latePruning(node.l_son, smaller_v)
    node.r_son = latePruning(node.r_son, bigger_equale_v)

    err_prune = 0
    err_no_prune = 0
    for idx in v:
        v_idx_label = id_three.test_group_dict['diagnosis'][idx]
        err_prune += evaluate(v_idx_label, node.label)
        err_no_prune += evaluate(v_idx_label, id3.classifier(idx,node))

    if err_prune < err_no_prune:
        node.f = None
        node.l_son = None
        node.r_son = None

    return node


def main():
    # because my optimal m was 1, it means no prune, i use the algorithem with no pruning
    id3.fit(id_three.train_row_indices)
    id3.decision_tree = latePruning(id3.decision_tree, id_three.test_row_indices)
    print(id3.predictLoss(id_three.test_row_indices))


if __name__ == "__main__":
    main()
