from torch.utils.data import Dataset
from utilities import Tokenizer
from sst_reader import sst_reader

class SSTAncestor(Dataset):
    def __init__(self, mode='train', tokenizer=None, granularity=3,
                 threshold=1, level=1, subtrees=False):

        if tokenizer:
            self.tokenizer = Tokenizer(tokenizer)

        if mode == 'train':
            self.data = list(sst_reader(train=True, dev=False, test=False, level=level,
                                        subtrees=subtrees))

        elif mode == 'val':
            self.data = list(sst_reader(train=False, dev=True, test=False, level=level,
                                        subtrees=subtrees))

        elif mode == 'test':
             self.data = list(sst_reader(train=False, dev=False, test=True, level=level,
                                         subtrees=subtrees))

        if granularity == 6:
            label_to_id = {}
            label_to_id['0'] = 0
            label_to_id['1'] = 1
            label_to_id['2'] = 2
            label_to_id['3'] = 3
            label_to_id['4'] = 4
            label_to_id['None'] = 5

        elif granularity == 3:
            label_to_id = {}
            label_to_id['None'] = 3
            label_to_id['0'] = 0
            label_to_id['1'] = 0
            label_to_id['2'] = 2
            label_to_id['3'] = 1
            label_to_id['4'] = 1

        data_list = []
        for i in self.data:
            if len(i['text'].split()) >= threshold:
                label = [label_to_id[x] for x in i['label']]
                base = [label_to_id[x] for x in i['base']]
                text = i['text'].split()
                # map to labels
                orig_to_tok_map = []

                if tokenizer:
                    tokens = []
                    # map to labels
                    for word in text:
                        orig_to_tok_map.append(len(tokens))
                        tokens.extend(self.tokenizer.tokenize(word))

                    text = tokens

                data_list.append({'text': text, 'label': label, 'map': orig_to_tok_map,
                                  'base': base})

        self.data = data_list
        del data_list

        if granularity == 3:
            for i in range(len(self.data)):
                del_idxs = []
                for j in range(len(self.data[i]['label'])):
                    if self.data[i]['label'][j] == 2:
                        del_idxs.append(j)

                self.data[i]['label'] = [self.data[i]['label'][x] for x in \
                    range(len(self.data[i]['label'])) if x not in del_idxs]
                self.data[i]['map'] = [self.data[i]['map'][x] for x in \
                    range(len(self.data[i]['map'])) if x not in del_idxs]
                self.data[i]['base'] = [self.data[i]['base'][x] for x in \
                    range(len(self.data[i]['base'])) if x not in del_idxs]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data[idx]['text']
        label = self.data[idx]['label']
        base = self.data[idx]['base']
        orig_to_tok_map = self.data[idx]['map']

        return {'text': sentence, 'label': label, 'map': orig_to_tok_map, 'base': base}
