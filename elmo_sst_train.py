import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import time

from allennlp.modules.elmo import Elmo, batch_to_ids
from allennlp.commands.elmo import ElmoEmbedder
from probing_models import LinearSST, NonLinearSST
from sst_dataset import SST
from torch.utils.data import DataLoader
from utilities import Visualizations, print_loss


# TODO Move all tokenization to the dataset class


def get_args():
    """get input arguments"""
    parser = argparse.ArgumentParser(description="train")

    parser.add_argument('--config', default='original', type=str)
    parser.add_argument('--epochs', default=60, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--save', default=True, type=bool)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--granularity', default=5, type=int)
    parser.add_argument('--layer', default='1', type=int)

    return parser.parse_args()


class ElmoCollate:
    def __call__(self, batch):
        # each element in "batch" is a dict {text:, label:}
        lengths = [len(x['text']) for x in batch]
        sentences = [x['text'] for x in batch]
        labels = [x['label'] for x in batch]

        return sentences, labels, lengths


def train(train_loader, model, criterion, optimizer, embedder, layer, granularity):
    running_loss = 0
    running_acc = 0
    iteration_count = 0
    num_samples = 0
    start = time.time()
    model.train()

    for idx, sample in enumerate(train_loader):
        sentences, labels, lengths = sample

        # Get scalar mix of all layers for elmo representation
        if layer == -1:
            sentences = batch_to_ids(sentences).cuda()
            sentences = embedder(sentences)['elmo_representations'][0]

            word_batch = []
            word_labels = []
            for i in range(len(lengths)):
                for j in range(lengths[i]):
                    word_batch.append(sentences[i][j])
                    word_labels.append(labels[i])

        # Get activations from individual layers
        else:
            sentences, _ = embedder.batch_to_embeddings(sentences)

            word_batch = []
            word_labels = []
            for i in range(len(lengths)):
                for j in range(lengths[i]):
                    word_batch.append(sentences[i][layer][j])
                    word_labels.append(labels[i])

        num_samples += len(word_labels)
        word_batch = torch.stack(word_batch, dim=0)
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

    return running_loss / iteration_count, accuracy, train_time


def test(test_loader, model, criterion, embedder, layer, granularity):
    running_loss = 0
    running_acc = 0
    iteration_count = 0
    num_samples = 0
    start = time.time()
    model.eval()

    with torch.no_grad():
        for idx, sample in enumerate(test_loader):
            sentences, labels, lengths = sample

            # Get scalar mix of all layers for elmo representation
            if layer == -1:
                sentences = batch_to_ids(sentences).cuda()
                sentences = embedder(sentences)['elmo_representations'][0]

                word_batch = []
                word_labels = []
                for i in range(len(lengths)):
                    for j in range(lengths[i]):
                        word_batch.append(sentences[i][j])
                        word_labels.append(labels[i])

            # Get activations from individual layers
            else:
                sentences, _ = embedder.batch_to_embeddings(sentences)

                word_batch = []
                word_labels = []
                for i in range(len(lengths)):
                    for j in range(lengths[i]):
                        word_batch.append(sentences[i][layer][j])
                        word_labels.append(labels[i])

            num_samples += len(word_labels)

            word_batch = torch.stack(word_batch, dim=0)
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

    return running_loss / iteration_count, accuracy, test_time


def main():
    # Collect input arguments & hyperparameters
    args = get_args()
    save = args.save
    config = args.config
    batch_size = args.batch_size
    learn_rate = args.lr
    epochs = args.epochs
    granularity = args.granularity
    layer = args.layer
    vis = Visualizations()

    # Load pretrained ELMo model from files
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

    # set tokenizer to process dataset with
    tokenizer = 'moses'
    # Load SST datasets into memory
    print("Processing datasets..")
    train_data = SST(mode='train', subtrees=True, granularity=granularity,
                     tokenizer=tokenizer)
    val_data = SST(mode='val', subtrees=False, granularity=granularity,
                   tokenizer=tokenizer)
    test_data = SST(mode='test', subtrees=True, granularity=granularity,
                    tokenizer=tokenizer)
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

    model = NonLinearSST(embedding_dim=1024, hidden_dim=1024, granularity=granularity)
    # model = LinearSST(embedding_dim=1024, granularity=granularity)
    model = model.cuda()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)

    # track best test accuracy for model
    best_acc = 0
    for epoch in range(epochs):
        loss, acc, time = train(train_data, model, criterion, optimizer, embedder,
                                layer, granularity)
        val_loss, val_acc, val_time = test(val_data, model, criterion, embedder,
                                           layer, granularity)
        test_loss, test_acc, test_time = test(test_data, model, criterion, embedder,
                                              layer, granularity)

        # plot epoch stats
        vis.plot_loss(loss, epoch, 'train')
        vis.plot_loss(val_loss, epoch, 'val')
        vis.plot_loss(test_loss, epoch, 'test')

        if test_acc > best_acc:
            best_acc = test_acc
            # printout epoch stats
            print_loss(epoch, 'train', loss, acc, time)
            print_loss(epoch, 'val  ', val_loss, val_acc, val_time)
            print_loss(epoch, 'test ', test_loss, test_acc, test_time)
            print("")

            if save is True:
                savename = 'models/sst/elmo_' + config + '_' + str(layer) + '_sst-' + str(granularity) + '.pt'
                torch.save(model.state_dict(), savename)


if __name__ == '__main__':
    main()
