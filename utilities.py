import visdom

from allennlp.modules.elmo import batch_to_ids
from datetime import datetime
from sacremoses import MosesTokenizer # pylint: disable=import-error
from pytorch_transformers import BertTokenizer



class Visualizations:
    def __init__(self, env_name='main'):
        if env_name is None:
            env_name = str(datetime.now().strftime("%d-%m %Hh%M"))
        self.env_name = env_name
        self.vis = visdom.Visdom(env=self.env_name)
        self.loss_win = None

    def plot_loss(self, loss, step, name):
        self.loss_win = self.vis.line(
            [loss],
            [step],
            win=self.loss_win,
            update='append' if self.loss_win else None,
            name=name,
            opts=dict(
                xlabel='Epoch',
                ylabel='Loss',
                title='Loss (mean)',
            )
        )


def print_loss(epoch, mode, loss, acc, time):
    print(f"Epoch {epoch + 1} {mode}: Loss: {loss:.3f}, Acc: {acc:.3f}, "
          f"Time: {time:.2f}")


class Tokenizer:
    def __init__(self, mode):
        self.mode = mode

        if self.mode is 'moses':
            self.tokenizer = MosesTokenizer()

        elif self.mode is 'bert':
            self.tokenizer = BertTokenizer.from_pretrained(
                'bert-base-cased', do_lower_case=False)

    def tokenize(self, text):
        if self.mode is 'moses':
            text = self.tokenizer.tokenize(text, escape=False)

        elif self.mode is 'bert':
            text = self.tokenizer.encode(text)

        return text


class PadSequence:
    def __call__(self, batch):
        # each element in "batch" is a dict {text:, label:}
        padded = batch_to_ids([x['text'][0] for x in batch])

        lengths = [len(x['text'][0]) for x in batch]
        labels = [x['label'] for x in batch]
        labels = [x['label'] for x in batch]

        return padded, lengths, labels


