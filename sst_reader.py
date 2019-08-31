import io
from torchnlp.datasets import smt_dataset
from torch.utils.data import Dataset
# from utilities import Tokenizer
from nltk.tree import Tree, ParentedTree




filename = 'sst/trees/dev.txt'
with io.open(filename, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                tree = Tree.fromstring(line)
                parent_tree = ParentedTree.convert(tree)
                for i in parent_tree.subtrees():
                    print("Child: ", i)
                    print("Parent: ", i.parent())
                    if i.parent() != None:
                        print(i.label(), i.parent().label())
                    else:
                        print(i.label(), None)
                    print("")

                break

def sst_reader(path='sst/', train=True, dev=False, test=False, fine_grained=True,
               subtrees=False):

    if train is True:
        filename = 'sst/trees/train.txt'
    elif dev is True:
        

