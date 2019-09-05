import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import time

from allennlp.modules.elmo import Elmo, batch_to_ids
from allennlp.commands.elmo import ElmoEmbedder
from probing_models import LinearSST, NonLinearSST
from sst_ancestor_dataset import SSTAncestor
from torch.utils.data import DataLoader
from utilities import Visualizations, print_loss
from torch.nn.utils.rnn import pad_sequence


def get_args():
    """get input arguments"""
    parser = argparse.ArgumentParser(description="train")

    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--save', default=False, type=bool)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--config', default='large', type=str)
    parser.add_argument('--layer', default='2', type=int)
    parser.add_argument('--level', default='0', type=int)
    parser.add_argument('--granularity', default='6', type=int)
    parser.add_argument('--subtrees', default=False, type=bool)

    return parser.parse_args()


class ElmoCollate:
    def __call__(self, batch):
        # each element in "batch" is a dict {text:, label:}
        lengths = [len(x['text']) for x in batch]
        sentences = [x['text'] for x in batch]
        labels = [x['label'] for x in batch]
        base = [x['base'] for x in batch]
        tok_to_label = [x['map'] for x in batch]

        return sentences, labels, lengths, tok_to_label, base


def train(train_loader, model, criterion, optimizer, embedder, layer, granularity):
    running_loss = 0
    running_acc = 0
    iteration_count = 0
    num_samples = 0
    start = time.time()
    model.train()

    for idx, sample in enumerate(train_loader):
        sentences, labels, lengths, tok_to_label, _ = sample
        # Get all intermediate layer embeddings from elmo
        sentences, _ = embedder.batch_to_embeddings(sentences)

        # Get activations from individual layers
        word_batch = []
        word_labels = []
        for i in range(len(lengths)):
            for j in range(len(tok_to_label[i])):
                word_batch.append(sentences[i][layer][tok_to_label[i][j]])
                word_labels.append(labels[i][j])

        num_samples += len(word_labels)
        word_batch = torch.stack(word_batch, dim=0).cuda()
        labels = torch.LongTensor(word_labels).cuda()

        # zero gradients
        model.zero_grad()
        scores = model(word_batch)
        scores = scores.view(-1, granularity)
        labels = labels.view(-1)

        # get accuracy scores
        for idx, i in enumerate(scores):
            _, pos = i.max(0)
            if pos == labels[idx]:
                running_acc += 1

        loss = criterion(scores, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        iteration_count += 1

    train_time = time.time() - start
    accuracy = (running_acc / num_samples) * 100
    loss = running_loss / iteration_count

    return loss, accuracy, train_time

def test(test_loader, model, criterion, embedder, layer, granularity):
    running_loss = 0
    running_acc = 0
    iteration_count = 0
    num_samples = 0
    start = time.time()
    model.eval()

    with torch.no_grad():
        for idx, sample in enumerate(test_loader):
            sentences, labels, lengths, tok_to_label, _ = sample

            # Get all intermediate layer embeddings from elmo
            sentences, _ = embedder.batch_to_embeddings(sentences)

            # Get activations from individual layers
            word_batch = []
            word_labels = []
            for i in range(len(lengths)):
                for j in range(len(tok_to_label[i])):
                    word_batch.append(sentences[i][layer][tok_to_label[i][j]])
                    word_labels.append(labels[i][j])

            num_samples += len(word_labels)
            word_batch = torch.stack(word_batch, dim=0).cuda()
            labels = torch.LongTensor(word_labels).cuda()
            scores = model(word_batch)
            scores = scores.view(-1, granularity)
            labels = labels.view(-1)

            # get accuracy scores
            for idx, i in enumerate(scores):
                _, pos = i.max(0)
                if pos == labels[idx]:
                    running_acc += 1

            loss = criterion(scores, labels)
            running_loss += loss.item()
            iteration_count += 1

    test_time = time.time() - start
    accuracy = (running_acc / num_samples) * 100
    loss = running_loss / iteration_count

    return loss, accuracy, test_time

def main():
    # Collect input arguments & hyperparameters
    args = get_args()
    config = args.config
    batch_size = args.batch_size
    learn_rate = args.lr
    epochs = args.epochs
    save = args.save
    granularity = args.granularity
    layer = args.layer
    level = args.level
    subtrees = args.subtrees

    savename = f"models/sst/elmo_{config}_{layer}_sst-{granularity}_{level}"
    print(savename)

    # Start visdom environment
    vis = Visualizations()

    # set tokenizer to process dataset with
    tokenizer = 'moses'
    # Load SST datasets into memory
    print("Processing datasets..")
    train_data = SSTAncestor(mode='train', tokenizer=tokenizer, granularity=granularity,
                             threshold=4, level=level, subtrees=subtrees)
    val_data = SSTAncestor(mode='val', tokenizer=tokenizer, granularity=granularity,
                             threshold=4, level=level, subtrees=False)
    test_data = SSTAncestor(mode='test', tokenizer=tokenizer, granularity=granularity,
                             threshold=4, level=level, subtrees=False)

    # Printout dataset stats
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Testing samples: {len(test_data)}")

    # Load datasets into torch dataloaders
    train_data = DataLoader(train_data, batch_size=batch_size, pin_memory=True,
                            shuffle=True, num_workers=6, collate_fn=ElmoCollate())

    val_data = DataLoader(val_data, batch_size=batch_size, pin_memory=True,
                          shuffle=False, num_workers=6, collate_fn=ElmoCollate())

    test_data = DataLoader(test_data, batch_size=batch_size, pin_memory=True,
                           shuffle=False, num_workers=6, collate_fn=ElmoCollate())

    # Load contextualizer model (ELMo)
    if config == 'original':
        options = "elmo/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weights = "elmo/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

    elif config == 'large':
        options = "elmo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
        weights = "elmo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"

    if layer == -1:
        embedder = Elmo(options, weights, 1, dropout=0).cuda()
    else:
        embedder = ElmoEmbedder(options_file=options, weight_file=weights, cuda_device=0)

    # for sentence level prediction, there are no instances of the "None" class.
    if level == 0:
        granularity -= 1

    # initialize probing model
    model = LinearSST(embedding_dim=1024, granularity=granularity)
    # model = NonLinearSST(embedding_dim=1024, hidden_dim=1024, granularity=granularity)
    model = model.cuda()

    # set loss function and optimizer
    criterion = nn.NLLLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)

    best_acc = 0
    for epoch in range(epochs):
        # Output = (loss, acc, time)
        train_out = train(train_data, model, criterion, optimizer, embedder, layer,
                          granularity)

        val_out = test(val_data, model, criterion, embedder, layer, granularity)
        test_out = test(test_data, model, criterion, embedder, layer, granularity)

        if test_out[1] > best_acc:
            best_acc = test_out[1]
            # printout epoch stats
            print("")
            print_loss(epoch, 'train', train_out[0], train_out[1], train_out[2])
            print_loss(epoch, 'val  ', val_out[0], val_out[1], val_out[2])
            print_loss(epoch, 'test ', test_out[0], test_out[1], test_out[2])

            if save is True:
                torch.save(model.state_dict(), savename + ".pt")

        # plot epoch stats
        vis.plot_loss(train_out[0], epoch, 'train')
        vis.plot_loss(val_out[0], epoch, 'val')
        vis.plot_loss(test_out[0], epoch, 'test')

if __name__ == '__main__':
    main()
    