from allennlp.modules.elmo import batch_to_ids

class PadSequence:
    def __call__(self, batch):
        # each element in "batch" is a dict {text:, label:}
        padded = batch_to_ids([x['text'][0] for x in batch])

        lengths = [len(x) for x in padded]
        labels = [x['label'] for x in batch]
        labels = [x['label'] for x in batch]

        return padded, lengths, labels