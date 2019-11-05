import torch
import os
from embedders import TokenEmbedder
from torch.utils.data import Dataset
from utilities import Tokenizer
from sst_reader import sst_reader


class SSTAncestor(Dataset):
    def __init__(self, mode='train', tokenizer=None, embed=None, granularity=3,
                 threshold=3, level=-1, subtrees=False, layer=0, save=True, load=True):

        self.level = level
        self.config = f'sst/saved/sst{granularity}_{mode}_{embed}-{layer}.pt'

        if load and os.path.exists(self.config):
            self.data = torch.load(self.config,
                                   map_location=lambda storage, loc: storage.cuda(0))

        else:
            if tokenizer:
                tokenizer = Tokenizer(tokenizer)

            if embed:
                embedder = TokenEmbedder(embed)

            if mode == 'train':
                self.data = list(sst_reader(train=True, dev=False, test=False,
                                            level=level, subtrees=subtrees))

            elif mode == 'val':
                self.data = list(sst_reader(train=False, dev=True, test=False,
                                            level=level, subtrees=subtrees))

            elif mode == 'test':
                self.data = list(sst_reader(train=False, dev=False, test=True,
                                            level=level, subtrees=subtrees))

            if granularity == 6:
                label_to_id = {
                    '0': 0,
                    '1': 1,
                    '2': 2,
                    '3': 3,
                    '4': 4,
                    None: 5
                }

            elif granularity == 3:
                label_to_id = {
                    '0': 0,
                    '1': 0,
                    '2': 2,
                    '3': 1,
                    '4': 1,
                    None: 3
                }

            data_list = []
            for i in self.data:
                text = i['text'].split()

                if len(text) >= threshold:
                    labels = []
                    for l in i['label']:
                        labels.append([label_to_id[x] for x in l])

                    # perform further tokenization on words
                    if tokenizer:
                        orig_to_tok_map = []
                        tokens = []
                        for word in text:
                            orig_to_tok_map.append(len(tokens))
                            tokens.extend(tokenizer.tokenize(word))

                        text = tokens

                    if embed:
                        # tokens: [layer, timestep, embed_dim], device=cuda
                        tokens = embedder.embed(text, layer)
                        labels = torch.LongTensor(labels).cuda()
                        label_count = 0
                        for i in range(len(tokens[0])):
                            if not tokenizer or i in orig_to_tok_map:
                                token_list = []
                                for j in range(len(tokens)):
                                    token_list.append(tokens[j][i])
                                token_list = torch.stack(token_list).half()
                                data_list.append([token_list, labels[label_count, :]])
                                label_count += 1

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

            if save == True:
                for i in range(len(self.data[0][0])):
                    words = torch.stack([x[0][i] for x in self.data]).half()
                    labels = torch.stack([x[1] for x in self.data])
                    data = [[words[i], labels[i]] for i in range(len(words))]
                    save = f'sst/saved/sst{granularity}_{mode}_{embed}-{i}.pt'
                    torch.save(data, save)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        word = self.data[idx][0]
        label = self.data[idx][1][self.level+1]

        return word, label