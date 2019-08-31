import io
from torchnlp.datasets import smt_dataset
from torch.utils.data import Dataset
# from utilities import Tokenizer
from nltk.tree import Tree, ParentedTree


def sst_reader(train=True, dev=False, test=False, level='1'):

    if train is True:
        filename = 'sst/trees/train.txt'
    elif dev is True:
        filename = 'sst/trees/dev.txt'
    elif test is True:
        filename = 'sst/trees/test.txt'

    data = []
    with io.open(filename, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                tree = Tree.fromstring(line)
                parent_tree = ParentedTree.convert(tree)

                for subtree in parent_tree.subtrees():
                    if level == 1:
                        if subtree.parent() == None:
                            data.append({
                                'text': ' '.join(subtree.leaves()),
                                'label': "None"
                            })
                        else:
                            data.append({
                                'text': ' '.join(subtree.leaves()),
                                'label': subtree.parent().label()
                            })

                    elif level == 2:
                        if subtree.parent() == None:
                            data.append({
                                'text': ' '.join(subtree.leaves()),
                                'label': "None"
                            })
                        elif subtree.parent().parent() == None:
                            data.append({
                                'text': ' '.join(subtree.leaves()),
                                'label': "None"
                            })

                        else:
                            data.append({
                                'text': ' '.join(subtree.leaves()),
                                'label': subtree.parent().parent().label()
                            })

    return data


