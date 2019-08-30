
# **Linear Probing**

Results collected from a simple linear probe layer trained on top of frozen contextualised word representations output from all intermediate layers of two state of the art contextualizers (BERT, ELMo). Much work has been done to analyse the internal representations and geometry of pretrained language models, in terms of syntactic and semantic subspaces. However, little has been done to determine how architecture-specific internal representations are mapped to a single contextual word embedding.

## **To-do List**

- Tabulate POS results
- Add scalar mix results
- Perform Binary Sentiment Classification
- Implement early stopping with patience
- Experiment will full treebank phrases for training for additional results
- Run all pretrained representations on non-linear classifier to determine the performance being constrained by using linear models.
- Check for other papers looking at sentence level sentiment prediction using single word representations
- Check oslo sentiment paper for dataset containing different linguist phenomena
- Qualitative assessment for correct classifications which override/negate the ground sentiment of the word with no context.


## **Linear Probing Results**

Results for linear probing tasks for all pretrained representations used in the study. All linear models are trained ontop of frozen contextualizers for 60 epochs using the Adam optimizer with a learning rate of 0.0001. Results from the epoch with the best accuracy are used.

## Pretrained Contextualizers

The following pretrained contextualizer models are used in this work.

**BERT (base, cased)**: 12-layer, 768-hidden, 12-heads, 110M parameters.
Trained on cased English text: Wikipedia (~2.5B words) + BookCorpus (~800M words))

**ELMo (original)**: 2-layer (+ non-contextual embedding layer), 1024-representation, 93.6M parameters. Trained on cased English text: Google 1 Billion words (~800M tokens)

**ELMo (5.5B)**: 2-layer (+ non-contextual embedding layer), 1024-representation, 93.6M parameters. Trained on cased English text: 5.5B tokens consisting of Wikipedia (1.9B) and all of the monolingual news crawl data from WMT 2008-2012 (3.6B)

## Non-contextualised benchmark

**GLoVe (840B.300d)**: (GLoVe description)

## **Sentiment Analysis (root sentences)**

We perform sentence-level sentiment analysis using single word representations contextualised on full sentences. We experiment with both fine-grained (5 classes) and binary, using the Stanford Sentiment Treebank (SST).

| Contextualizer              | SST-5    | SST-2    |
|:--------------------------- |:---------|:--------:|
|BERT (base, cased), layer 12 | 39.180   | 74.011
|BERT (base, cased), layer 11 | 37.628   | 72.173
|BERT (base, cased), layer 10 | 36.734   | 70.489
|BERT (base, cased), layer 9  | 35.364   | 70.025
|BERT (base, cased), layer 8  | 35.793   |
|BERT (base, cased), layer 7  | 34.464   |
|BERT (base, cased), layer 6  | 33.716   |
|BERT (base, cased), layer 5  | 31.384*  |
|BERT (base, cased), layer 4  | 31.984*  |
|BERT (base, cased), layer 3  | 31.465*  |
|BERT (base, cased), layer 2  | 31.407*  |
|BERT (base, cased), layer 1  | 31.253*  |
|BERT (base, cased), layer 0  | 24.145*  |
|||
|ELMo (original), layer 2     | 39.461   |
|ELMo (original), layer 1     | 38.206   |
|ELMo (original), layer 0     | 29.097   |
|||
|ELMo (5.5B), layer 2         | 40.151   | 77.013
|ELMo (5.5B), layer 1         | 38.701   | 75.042
|ELMo (5.5B), layer 0         | 29.052   | 57.197
|||
| GLoVe *840B.300d)           | 28.808   | 57.721  |
*retest

## **Part of Speech**

| Model     | PTB     | Parent   | Grandparent |
|:---------:|:-------:|:--------:|:-----------:|
| GLoVe     |         |          |             |
| ELMo      |         |          |             |
| BERT      |         |          |             |

## Ancestor Sentiment Classification

For a sequence of contextualized word representations, each token is tasked with predicting the sentiment classification of it's parent, grandparent, or great-grandparent. The root sentiment classification (sentence-level) is removed, as this is already tested. For cases where the token doesn't have a grandparent or great-grandparent, the linear model is tasked to predict a "None" classification is label.
