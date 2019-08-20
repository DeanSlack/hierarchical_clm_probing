import visdom

from allennlp.modules.elmo import batch_to_ids
from datetime import datetime


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
                xlabel='Step',
                ylabel='Loss',
                title='Loss (mean)',
            )
        )


class NoPad:
    def __call__(self, batch):
        # each element in "batch" is a dict {text:, label:}
        sentences = [x['text'] for x in batch]
        labels = [x['label'] for x in batch]
        lengths = [len(x['text']) for x in batch]

        return sentences, labels, lengths


class PadSequence:
    def __call__(self, batch):
        # each element in "batch" is a dict {text:, label:}
        padded = batch_to_ids([x['text'][0] for x in batch])

        lengths = [len(x['text'][0]) for x in batch]
        labels = [x['label'] for x in batch]
        labels = [x['label'] for x in batch]

        return padded, lengths, labels