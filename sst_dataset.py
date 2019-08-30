from torchnlp.datasets import smt_dataset
from torch.utils.data import Dataset
from utilities import Tokenizer

class SST(Dataset):
    def __init__(self, mode='train', subtrees=False, embedder=None, tokenizer=None,
                 granularity=2, threshold=3):

        if granularity == 5:
            fine_grained = True
        else:
            fine_grained = False

        if tokenizer:
            self.tokenizer = Tokenizer(tokenizer)

        self.subtrees = subtrees

        if mode == 'train':
            self.data = list(smt_dataset('sst/', train=True, fine_grained=fine_grained,
                                         subtrees=self.subtrees))

        if mode == 'val':
            self.data = list(smt_dataset('sst/', train=False, dev=True,
                                         fine_grained=fine_grained,
                                         subtrees=self.subtrees))

        if mode == 'test':
            self.data = list(smt_dataset('sst/', train=False, test=True,
                                         fine_grained=fine_grained,
                                         subtrees=self.subtrees))

        if fine_grained is True:
            label_to_id = {}
            label_to_id['very negative'] = 0
            label_to_id['negative'] = 1
            label_to_id['neutral'] = 2
            label_to_id['positive'] = 3
            label_to_id['very positive'] = 4

        else:
            label_to_id = {}
            label_to_id['very negative'] = 0
            label_to_id['negative'] = 0
            label_to_id['neutral'] = 2
            label_to_id['positive'] = 1
            label_to_id['very positive'] = 1

        if self.subtrees == False:
            for i in self.data:
                i['label'] = label_to_id[i['label']]

                if tokenizer:
                    i['text'] = self.tokenizer.tokenize(i['text'])

        else:
            data_list = []
            count = 0
            for i in self.data:
                if len(i['text'].split()) >= threshold:
                    label = label_to_id[i['label']]

                    if tokenizer:
                        text = self.tokenizer.tokenize(i['text'])

                    else:
                        text = i['text']

                    data_list.append({'text': text, 'label': label})
                    count += 1

            self.data = data_list
            del data_list

        del_idxs = []
        if fine_grained is False:
            for i in range(len(self.data)):
                if self.data[i]['label'] == 2:
                    del_idxs.append(i)

        self.data = [self.data[x] for x in range(len(self.data)) if x not in del_idxs]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data[idx]['text']
        label = self.data[idx]['label']

        return {'text': sentence, 'label': label}
