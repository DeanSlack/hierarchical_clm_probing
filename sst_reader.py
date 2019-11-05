import io
from nltk.tree import Tree, ParentedTree


def sst_reader(train=True, dev=False, test=False, level='1', subtrees=False):

    if train is True:
        filename = 'sst/trees/train.txt'
    elif dev is True:
        filename = 'sst/trees/dev.txt'
    elif test is True:
        filename = 'sst/trees/test.txt'

    data = []
    with io.open(filename, encoding='utf-8') as f:
        if subtrees is False:
            for line in f:
                line = line.strip()
                tree = Tree.fromstring(line)
                ptree = ParentedTree.convert(tree)
                text = ' '.join(tree.leaves())
                parent_labels = []
                word_labels = []
                for idx, _ in enumerate(ptree.leaves()):
                    leafpos = ptree.leaf_treeposition(idx)
                    if level == -1:
                        parent_pos = 0
                    if level != -1:
                        parent_pos = leafpos[:-(level+1)]
                    if parent_pos == []:
                        parent_labels.append(None)
                    elif level == -1:
                        parent_labels.append([tree.label(),
                                              ptree[leafpos[:-(1)]].label(),
                                              ptree[leafpos[:-(2)]].label(),
                                              ptree[leafpos[:-(3)]].label(),
                                              ptree[leafpos[:-(4)]].label(),
                                              ptree[leafpos[:-(5)]].label(),
                                              ptree[leafpos[:-(6)]].label(),
                                              ptree[leafpos[:-(7)]].label(),
                                              ptree[leafpos[:-(8)]].label(),
                                              ptree[leafpos[:-(9)]].label(),
                                              ptree[leafpos[:-(10)]].label()])
                    else:
                        parent_labels.append(ptree[parent_pos].label())
                    # testing with sentence sentiment
                    word_labels.append(tree[leafpos[0]].label())

                data.append({'text': text, 'label': parent_labels, 'base': word_labels})

        else:
            for line in f:
                line = line.strip()
                tree = Tree.fromstring(line)
                tree = ParentedTree.convert(tree)

                for subtree in tree.subtrees():
                    if len(subtree.leaves()) >= 3:
                        text = ' '.join(subtree.leaves())
                        parent_labels = []
                        word_labels = []
                        for idx, _ in enumerate(subtree.leaves()):
                            leafpos = subtree.leaf_treeposition(idx)
                            if level == -1:
                                parent_pos = leafpos[0]
                            else:
                                parent_pos = leafpos[:-(level+1)]
                            if parent_pos == []:
                                parent_labels.append('None')
                            else:
                                parent_labels.append(subtree[parent_pos].label())
                            # testing with sentence sentiment
                            word_labels.append(subtree[leafpos[0]].label())

                        data.append({'text': text, 'label': parent_labels, 'base': word_labels})

    return data
