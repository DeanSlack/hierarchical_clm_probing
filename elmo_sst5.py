import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import time

from allennlp.modules.elmo import Elmo, batch_to_ids
from allennlp.commands.elmo import ElmoEmbedder
from probing_models import LinearSST
from sst_dataset import SST
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utilities import PadSequence, Visualizations, NoPad

#TODO rename file to elmo_sst.py


vis = Visualizations()


def get_args():
    """get input arguments"""
    parser = argparse.ArgumentParser(description="train")

    parser.add_argument('--model', default='', type=str)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--save', default=False, type=bool)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--fine_grained', default=True, type=bool)
    parser.add_argument('--layer', default='1', type=int)

    return parser.parse_args()


def train(train_loader, model, criterion, optimizer, embedder, layer, iterations):
    global vis
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
        scores = scores.view(-1, 5)
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
        iterations += 1

        if iterations % 10 == 0:
            vis.plot_loss(running_loss/iteration_count, iterations, 'train')

    train_time = time.time() - start
    accuracy = (running_acc / num_samples) * 100

    return running_loss / iteration_count, accuracy, train_time, iterations


def test(test_loader, model, criterion, embedder, layer, iterations):
    global vis
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
            scores = scores.view(-1, 5)
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

    vis.plot_loss(running_loss/iteration_count, iterations, 'test')

    return running_loss / iteration_count, accuracy, test_time


def main():
    # Collect input arguments & hyperparameters
    args = get_args()
    save = args.save
    batch_size = args.batch_size
    learn_rate = args.lr
    epochs = args.epochs
    fine_grained = args.fine_grained
    layer = args.layer

    if fine_grained is True:
        granularity = 5
    else:
        granularity = 2

    # Load pretrained ELMo model from files
    options = "elmo/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weights = "elmo/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

    if layer == -1:
        elmo = Elmo(options, weights, 1, dropout=0).cuda()
    else:
        elmo = ElmoEmbedder(options_file=options, weight_file=weights, cuda_device=0)

    # Load SST datasets into memory
    print("Processing datasets..")
    train_data = SST(mode='train', subtrees=True)
    val_data = SST(mode='val')
    test_data = SST(mode='test')
    # Printout dataset stats
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Testing samples: {len(test_data)}")

    # Load datasets into torch dataloaders
    train_data = DataLoader(train_data, batch_size=batch_size, pin_memory=True,
                            shuffle=True, num_workers=4, collate_fn=NoPad())

    val_data = DataLoader(val_data, batch_size=batch_size, pin_memory=False,
                          shuffle=False, num_workers=4, collate_fn=NoPad())

    test_data = DataLoader(test_data, batch_size=batch_size, pin_memory=False,
                           shuffle=False, num_workers=4, collate_fn=NoPad())

    model = LinearSST(embedding_dim=1024, granularity=granularity)
    model = model.cuda()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)

    # track best test accuracy for model
    best_test_acc = 0
    iters = 0

    for epoch in range(epochs):
        loss, acc, time, iters = train(train_data, model, criterion, optimizer, elmo,
                                       layer, iters)

        print(f"Epoch {epoch + 1} Train: Loss: {loss:.3f}, Acc: {acc:.3f}, \
            Time: {time:.2f}")

        # val_loss, val_acc, val_time = test(val_data, model, criterion, elmo, layer,
        #                                    iters)
        # print(f"Epoch {epoch + 1} Valid: Loss: {val_loss:.3f}, Acc: {val_acc:.3f}, \
        #     Time: {val_time:.2f}")

        test_loss, test_acc, test_time = test(test_data, model, criterion, elmo, layer,
                                              iters)
        print(f"Epoch {epoch + 1} Test:  Loss: {test_loss:.3f}, \
            Acc: {test_acc:.3f}, Time: {test_time:.2f}")

        if test_acc > best_test_acc and save is True:
            print("Saving model..")
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'sst_model_2.pt')

        print("")


if __name__ == '__main__':
    main()