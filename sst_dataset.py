from allennlp.modules.elmo import batch_to_ids
from torchnlp.datasets import smt_dataset
from torch.utils.data import DataLoader, Dataset


class SST(Dataset):
    def __init__(self, mode='train', subtrees=False, embedder=None, split=True):

        self.embedder = embedder
        if self.embedder:
            self.embedder = self.embedder.cuda()

        self.subtrees = subtrees

        if mode == 'train':
            self.data = list(smt_dataset('sst/', train=True, fine_grained=True,
                                         subtrees=self.subtrees))

        if mode == 'val':
            self.data = list(smt_dataset('sst/', train=False, dev=True,
                                         fine_grained=True, subtrees=self.subtrees))

        if mode == 'test':
            self.data = list(smt_dataset('sst/', train=False, test=True,
                                         fine_grained=True, subtrees=self.subtrees))

        label_to_id = {}
        label_to_id['very negative'] = 0
        label_to_id['negative'] = 1
        label_to_id['neutral'] = 2
        label_to_id['positive'] = 3
        label_to_id['very positive'] = 4

        if self.subtrees == False:
            for i in self.data:
                i['label'] = label_to_id[i['label']]
                if split == True:
                    i['text'] = i['text'].split()
                else:
                    i['text'] = i['text']
                if self.embedder:
                    i['text'] = self.embedder(
                        batch_to_ids(i['text']).cuda())['elmo_representations'][0][0]

        else:
            del_list = []
            count = 0
            for i in self.data:
                if len(i['text'].split()) > 3:
                    label = label_to_id[i['label']]
                    text = i['text'].split()
                    if self.embedder:
                        text = self.embedder(batch_to_ids(text).cuda())['elmo_representations'][0][0]

                    del_list.append({'text': text, 'label': label})
                    count += 1
            self.data = del_list

        del self.embedder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data[idx]['text']
        label = self.data[idx]['label']

        return {'text': sentence, 'label': label}
