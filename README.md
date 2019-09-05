
# **Linear Probing on Contextual Word Representations**

Results collected from a simple linear probe layer trained on top of frozen contextualised word representations output from all intermediate layers of two state of the art contextualizers (BERT, ELMo). Much work has been done to analyse the internal representations and geometry of pretrained language models, in terms of syntactic and semantic subspaces. However, little has been done to determine how architecture-specific internal representations are mapped to a single contextual word embedding. To assess the level of hierarchical and contextual information present in an embedding, we use sequence-level classification tasks (which are typically solved by inputting the entire sequence into a classifier) and constrain the input to a single embedding contextualised on the full sequence.

## **To-do List**

- Calculate average sentiment value change across depths of tree as a baseline
- Tabulate POS results
- Add scalar mix results
- Check different tokenization methods
- Which model encodes more local sentiment, and which more global? (Use performance of parent, grandparent, etc)
- Rerun all multiple times, use average of best per run + standard deviation
- Experiment using full training set, with higher sentence length threshold
- Implement early stopping with patience
- Experiment will full treebank phrases for training for additional results
- Run all pretrained representations on non-linear classifier to determine the performance being constrained by using linear models.
- Check for other papers looking at sentence level sentiment prediction using single word representations
- Check oslo sentiment paper for dataset containing different linguist phenomena
- Qualitative assessment for correct classifications which override/negate the ground sentiment of the word with no context.
- Can we utilize more treebank structured datasets / tasks?
- Best way to visualize? Horizontal histogram plots of layerwise accuracy per model per task
- Use lower layers as input to BiLSMT sentence-level classifier

## **Linear Probing Results**

Results for linear probing tasks for all pretrained representations used in the study. All linear models are trained ontop of frozen contextualizers for 60 epochs using the Adam optimizer with a learning rate of 0.0001. Results from the epoch with the best accuracy are used.

## Pretrained Contextualizers

The following pretrained contextualizer models are used in this work.

**BERT (base, cased)**: 12-layer, 768-hidden, 12-heads, 110M parameters.
Trained on cased English text: Wikipedia (~2.5B words) + BookCorpus (~800M words))

**ELMo (original)**: 2-layer (+ non-contextual embedding layer), 1024-representation, 93.6M parameters. Trained on cased English text: Google 1 Billion words (~800M tokens)

**ELMo (5.5B)**: 2-layer (+ non-contextual embedding layer), 1024-representation, 93.6M parameters. Trained on cased English text: 5.5B tokens consisting of Wikipedia (1.9B) and all of the monolingual news crawl data from WMT 2008-2012 (3.6B)

## Non-contextualised benchmark

**GloVe (840B.300d)**: (GloVe description)

## **Ancestor Sentiment Analysis (root-sentence training only)**

For a sequence of contextualized word representations, each token is tasked with predicting the sentiment classification of it's parent, grandparent, or great-grandparent. For cases where the token doesn't have a grandparent or great-grandparent, the linear model is tasked to predict a "None" classification is label. Additionally, we perform sentence-level (root) sentiment analysis using single word representations contextualised on full sentences. We experiment with both fine-grained (5 classes) and binary, using the Stanford Sentiment Treebank (SST).

| Contextualizer              | Root     | Leaf    |Parent  | GParent | GGParent |
|:--------------------------- |:--------:|:-------:|--------|:-------:|:--------:|
|BERT (base, cased), layer 12 | 40.957   | 88.325  |62.693  | 53.289  | 48.154
|BERT (base, cased), layer 11 | 40.594   | 88.950  |62.934  | 53.343  | 48.165
|BERT (base, cased), layer 10 | 36.734   | 89.328  |63.123  | 53.653  | 48.272
|BERT (base, cased), layer 9  | 35.364   | 89.760  |63.359  | 53.945  | 48.366
|BERT (base, cased), layer 8  | 35.793   | 90.204  |63.262  | 53.480  | 47.559
|BERT (base, cased), layer 7  | 34.464   | 90.926  |63.109  | 53.107  | 47.160
|BERT (base, cased), layer 6  | 33.716   | 91.314  |62.634  | 52.354  | 46.454
|BERT (base, cased), layer 5  | 31.384*  | 91.656  |62.039  | 51.306  | 45.552
|BERT (base, cased), layer 4  | 31.984*  | 91.923  |61.808  | 50.758  | 44.917
|BERT (base, cased), layer 3  | 31.465*  | 92.157  |61.472  | 50.286  | 44.291
|BERT (base, cased), layer 2  | 31.407*  | 92.383  |61.338  | 49.974  | 43.852
|BERT (base, cased), layer 1  | 31.253*  | 92.848  |60.648  | 48.907  | 42.508
|BERT (base, cased), layer 0  | 24.145*  | 92.829  |60.103  | 47.495  | 39.947
|||
|ELMo (original), layer 2     | 40.151   |         |        |         |
|ELMo (original), layer 1     | 38.701   |         |        |         |
|ELMo (original), layer 0     | 29.052   |         |        |         |
|||
|ELMo (5.5B), layer 2         | 42.193   | 90.565  | 54.639 |         |
|ELMo (5.5B), layer 1         | 38.701   |         | 54.727 | 54.668  |
|ELMo (5.5B), layer 0         | 31.298   |         |        |         |
|||
|GloVe (840B.300d)            | 28.808   |         |        |         |
*retest

| SST-2
|:--------:|
| 74.011
| 72.173
| 70.489
| 70.105
| 68.477
| 68.198
| 67.770
| 67.084
| 67.004
| 66.640
| 65.755
| 65.512
| 56.220
| 76.557
| 74.117
| 58.173
|||
| 77.013
| 75.788
| 58.255
|||
| 57.721

## **Part of Speech**

| Model     | PTB     | Parent   | Grandparent |
|:---------:|:-------:|:--------:|:-----------:|
| GLoVe     |         |          |             |
| ELMo      |         |          |             |
| BERT      |         |          |             |
