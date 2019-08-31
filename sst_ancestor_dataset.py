from torchnlp.datasets import smt_dataset
from torch.utils.data import Dataset
from utilities import Tokenizer
from sst_reader import sst_reader

class SSTAncestor(Dataset):
    def __init__(self, mode='train', tokenizer=None, granularity=3,
                 threshold=1, level=1):

        if granularity == 6:
            fine_grained = True
        else:
            fine_grained = False

        if tokenizer:
            self.tokenizer = Tokenizer(tokenizer)

        if mode == 'train':
            self.data = list(sst_reader(train=True, dev=False, test=False, level=level))

        elif mode == 'val':
            self.data = list(sst_reader(train=False, dev=True, test=False, level=level))

        elif mode == 'test':
             self.data = list(sst_reader(train=False, dev=False, test=True, level=level))


        if fine_grained is True:
            label_to_id = {}
            label_to_id['0'] = 0
            label_to_id['1'] = 1
            label_to_id['2'] = 2
            label_to_id['3'] = 3
            label_to_id['4'] = 4
            label_to_id['None'] = -1

        else:
            label_to_id = {}
            label_to_id['None'] = -1
            label_to_id['0'] = 0
            label_to_id['1'] = 0
            label_to_id['2'] = 2
            label_to_id['3'] = 1
            label_to_id['4'] = 1

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
