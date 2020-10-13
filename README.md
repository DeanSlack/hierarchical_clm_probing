
# **Linear Probing on Contextual Word Representations**

Results collected from a simple linear probe layer trained on top of frozen contextualised word representations output from all intermediate layers of two state of the art contextualizers (BERT, ELMo). Much work has been done to analyse the internal representations and geometry of pretrained language models, in terms of syntactic and semantic subspaces. However, little has been done to determine how architecture-specific internal representations are mapped to a single contextual word embedding. To assess the level of hierarchical and contextual information present in an embedding, we use sequence-level classification tasks (which are typically solved by inputting the entire sequence into a classifier) and constrain the input to a single embedding contextualised on the full sequence.

## **Linear Probing Results**

Results for linear probing tasks for all pretrained representations used in the study. All linear models are trained ontop of frozen contextualizers for 60 epochs using the Adam optimizer with a learning rate of 0.0001. Results from the epoch with the best accuracy are used.

## Pretrained Contextualizers

The following pretrained contextualizer models are used in this work.

**Reformer (base)**: 12-layer, 768-hidden, 12-heads, 125M parameters

**XLNet (base)**: 2-layer, 768-hidden, 12-heads, 110M parameters.
XLNet English model

**GPT2 (medium)**:24-layer, 1024-hidden, 16-heads, 345M parameters.
OpenAI’s Medium-sized GPT-2 English model

**GPT2 (base)**: 12-layer, 768-hidden, 12-heads, 117M parameters.
OpenAI GPT-2 English model

**BERT (large, cased)**: 24-layer, 1024-hidden, 16-heads, 340M parameters.
Trained on cased English text.

**BERT (base, cased)**: 12-layer, 768-hidden, 12-heads, 110M parameters.
Trained on cased English text: Wikipedia (~2.5B words) + BookCorpus (~800M words))

**ELMo (original)**: 2-layer (+ non-contextual embedding layer), 1024-representation, 93.6M parameters. Trained on cased English text: Google 1 Billion words (~800M tokens)

**ELMo (5.5B)**: 2-layer (+ non-contextual embedding layer), 1024-representation, 93.6M parameters. Trained on cased English text: 5.5B tokens consisting of Wikipedia (1.9B) and all of the monolingual news crawl data from WMT 2008-2012 (3.6B)

## Non-contextualised benchmark

**GloVe (840B.300d)**: (GloVe description)

## **Ancestor Tagging Avg (root-sentence training only)**

For a sequence of contextualized word representations, each token is tasked with predicting the sentiment classification of it's parent, grandparent, or great-grandparent. For cases where the token doesn't have a grandparent or great-grandparent, the linear model is tasked to predict a "None" classification label. Additionally, we perform sentence-level (root) sentiment analysis using single word representations contextualised on full sentences. We experiment with both fine-grained (5 classes) and binary, using the Stanford Sentiment Treebank (SST).

### Non-Transformer baselines

| Contextualizer              | Leaf    |Parent  | GParent | GGParent | Root     |
|:--------------------------- |:-------:|--------|:-------:|:--------:|:--------:|
|ELMo (original), layer 2     | 90.576  | 64.171  | 54.410  |**49.183**|**39.315**|
|ELMo (original), layer 1     |**91.598**|**64.060**|**54.205**| 48.319  | 38.101   |
|ELMo (original), layer 0     | 87.986  | 57.575  | 45.625  | 37.928  | 28.248   |
|||                                                                   |
|ELMo (5.5B), layer 2         | 90.133  | 63.860  | 54.766  | 49.567  | 41.181   |
|ELMo (5.5B), layer 1         | 91.131  | 64.244  | 54.818  | 49.621  | 39.341   |
|ELMo (5.5B), layer 0         | 91.326  | 59.647  | 47.692  | 40.117  | 29.307   |
|||                                                                   |
|GloVe (840B.300d)            | 90.274  | 60.278 | 47.526  | 39.963   | 28.808   |
*retest

### BERT (base, cased)

| Contextualizer              | Leaf    |Parent  | GParent | GGParent | Root     |
|:--------------------------- |:-------:|--------|:-------:|:--------:|:--------:|
|BERT (base, cased), layer 12 | 88.325  |62.693  | 53.289  | 48.154   | 41.155   |
|BERT (base, cased), layer 11 | 88.950  |62.934  | 53.343  | 48.165   | 40.978   |
|BERT (base, cased), layer 10 | 89.328  |63.123  | 53.653  | 48.272   | **41.256** |
|BERT (base, cased), layer 9  | 89.760  |**63.359**|**53.945**|**48.366**| 40.822   |
|BERT (base, cased), layer 8  | 90.204  |63.262  | 53.480  | 47.559   | 40.374   |
|BERT (base, cased), layer 7  | 90.926  |63.109  | 53.107  | 47.160   | 40.207   |
|BERT (base, cased), layer 6  | 91.314  |62.634  | 52.354  | 46.454   | 39.414   |
|BERT (base, cased), layer 5  | 91.656  |62.039  | 51.306  | 45.552   | 39.091   |
|BERT (base, cased), layer 4  | 91.923  |61.808  | 50.758  | 44.917   | 37.911   |
|BERT (base, cased), layer 3  | 92.157  |61.472  | 50.286  | 44.291   | 38.202   |
|BERT (base, cased), layer 2  | 92.383  |61.338  | 49.974  | 43.852   | 36.984   |
|BERT (base, cased), layer 1  | 92.848  |60.648  | 48.907  | 42.508   | 37.079   |
|BERT (base, cased), layer 0  | 92.829  |60.103  | 47.495  | 39.947   | 32.260   |

![alt-text](https://github.com/DeanSlack/lm_probing/blob/master/figures/bert_base.svg)

### GPT2 (base)

| Contextualizer              | Leaf    |Parent  | GParent | GGParent | Root     |
|:--------------------------- |:-------:|--------|:-------:|:--------:|:--------:|
|GPT2 (base), layer 0         | -       |61.54   | 49.79   | 43.88    | 29.32   |
|GPT2 (base), layer 1         | -       |61.57   | 50.49   | 46.06    | 31.40   |
|GPT2 (base), layer 2         | -       |61.63   | 50.63   | 46.70    | 31.59   |
|GPT2 (base), layer 3         | -       |61.65   | 51.41   | 47.37    | 32.46   |
|GPT2 (base), layer 4         | -       |61.94   | 52.13   | 48.08    | 32.09   |
|GPT2 (base), layer 5         | -       |61.99   | 52.65   | 48.62    | 32.55   |
|GPT2 (base), layer 6         | -       |62.03   | 53.38   | 48.91    | 33.33   |
|GPT2 (base), layer 7         | -       |62.29   | 52.99   | 48.39    | 33.00   |
|GPT2 (base), layer 8         | -       |62.07   | 52.75   | 49.29    | 33.99   |
|GPT2 (base), layer 9         | -       |61.90   | 52.74   | 48.90    | 34.26   |
|GPT2 (base), layer 10        | -       |61.42   | 51.89   | 48.11    | 33.92   |
|GPT2 (base), layer 11        | -       |60.55   | 50.45   | 46.74    | 33.16   |
|GPT2 (base), layer 12        | -       |**62.61**|**53.86**|**49.55**|**36.14**|

![alt-text](https://github.com/DeanSlack/lm_probing/blob/master/figures/gpt2.svg)

### XLNet (base)

| Contextualizer               | Leaf    |Parent  | GParent | GGParent | Root     |
|:-----------------------------|:-------:|--------|:-------:|:--------:|:--------:|
|XLNet (base), layer 0         | -       |59.45   | 46.28   | 38.31    | 29.02    |
|XLNet (base), layer 1         | -       |61.23   | 51.44   | 45.91    | 33.07    |
|XLNet (base), layer 2         | -       |62.35   | 53.96   | 48.67    | 34.30    |
|XLNet (base), layer 3         | -       |63.73   | 55.34   | 49.93    | 35.08    |
|XLNet (base), layer 4         | -       |64.62   | 56.54   | 51.65    | 37.66    |
|XLNet (base), layer 5         | -       |64.34   | 56.93   | 52.13    | 39.78    |
|XLNet (base), layer 6         | -       |64.92   | 57.44   | 52.77    | 39.71    |
|XLNet (base), layer 7         | -       |64.98   | 57.60   | 53.03    | 40.87    |
|XLNet (base), layer 8         | -       |**64.96**|**57.79**|**53.49**|**41.27**|
|XLNet (base), layer 9         | -       |64.66   | 57.05   | 53.26    | 41.17    |
|XLNet (base), layer 10        | -       |64.26   | 57.01   | 52.52    | 41.22    |
|XLNet (base), layer 11        | -       |63.68   | 56.54   | 51.54    | 40.48    |
|XLNet (base), layer 12        | -       |62.67   | 54.74   | 49.95    | 39.59    |

![alt-text](https://github.com/DeanSlack/lm_probing/blob/master/figures/xlnet.svg)

### Reformer (base)

| Contextualizer               | Leaf    |Parent  | GParent | GGParent | Root     |
|:-----------------------------|:-------:|--------|:-------:|:--------:|:--------:|
|Reformer (base), layer 0       | -       |61.65   | 49.96   | 43.78    | 29.43    |
|Reformer (base), layer 1       | -       |62.86   | 52.97   | 48.13    | 34.35    |
|Reformer (base), layer 2       | -       |62.79   | 54.05   | 49.89    | 34.09    |
|Reformer (base), layer 3       | -       |63.79   | 55.87   | 51.53    | 35.08    |
|Reformer (base), layer 4       | -       |64.53   | 56.78   | 52.60    | 36.37    |
|Reformer (base), layer 5       | -       |64.69   | 57.67   | 53.84    | 38.73    |
|Reformer (base), layer 6       | -       |**65.18**| 57.88   | 54.28    | 40.56    |
|Reformer (base), layer 7       | -       |64.89   |**58.14**|**54.57** | 40.29    |
|Reformer (base), layer 8       | -       |64.80   | 58.03   | 54.28    |**41.04**|
|Reformer (base), layer 9       | -       |64.49   | 57.72   | 53.85    | 39.54    |
|Reformer (base), layer 10      | -       |64.57   | 57.80   | 53.30    | 39.29    |
|Reformer (base), layer 11      | -       |63.91   | 57.52   | 53.23    | 39.86    |
|Reformer (base), layer 12      | -       |63.91   | 57.08   | 52.91    | 40.07    |

![alt-text](https://github.com/DeanSlack/lm_probing/blob/master/figures/reformer.svg)

### BERT (large, cased)

| Contextualizer              | Leaf    |Parent  | GParent | GGParent | Root     |
|:--------------------------- |:-------:|--------|:-------:|:--------:|:--------:|
|BERT (large, cased), layer 0 | 88.11   |60.22   | 48.69   | 43.19    | 28.55   |
|BERT (large, cased), layer 1 | 88.53   |61.23   | 50.91   | 46.30    | 34.45   |
|BERT (large, cased), layer 2 | 88.39   |61.30   | 51.07   | 46.70    | 34.20   |
|BERT (large, cased), layer 3 | 88.49   |61.34   | 51.08   | 47.01    | 33.88   |
|BERT (large, cased), layer 4 | 88.55   |61.88   | 51.95   | 47.55    | 33.08   |
|BERT (large, cased), layer 5 | 88.52   |62.00   | 52.21   | 47.95    | 33.03   |
|BERT (large, cased), layer 6 | 88.57   |62.22   | 52.70   | 48.41    | 32.78   |
|BERT (large, cased), layer 7 | 88.64   |62.75   | 53.04   | 48.44    | 32.73   |
|BERT (large, cased), layer 8 | 88.68   |62.80   | 53.31   | 48.88    | 33.32   |
|BERT (large, cased), layer 9 | 88.68   |63.04   | 53.89   | 49.46    | 34.36   |
|BERT (large, cased), layer 10 |88.68   |63.57   | 54.21   | 49.62    | 34.99   |
|BERT (large, cased), layer 11 |88.75   |63.69   | 54.57   | 50.47    | 35.79   |
|BERT (large, cased), layer 12 |**88.75**|64.41   | 55.26   | 51.21    | 36.45   |
|BERT (large, cased), layer 13 |88.58   |64.80   | 55.68   | 51.50    | 36.47   |
|BERT (large, cased), layer 14 |88.58   |65.12   | 56.65   | 51.45    | 36.59   |
|BERT (large, cased), layer 15 |88.58   |65.37   | 56.69   | 52.03    | 36.61   |
|BERT (large, cased), layer 16 |88.27   |65.55   | 57.18   | 52.61    | 37.81   |
|BERT (large, cased), layer 17 |88.21   |65.75   | 57.87   | 53.23    | 38.45   |
|BERT (large, cased), layer 18 |87.91   |**65.89**|**57.95**|**53.63**| 40.31   |
|BERT (large, cased), layer 19 |87.72   |65.43   | 57.86   | 53.27    | 40.64   |
|BERT (large, cased), layer 20 |87.59   |65.38   | 57.80   | 53.35    |**41.65**|
|BERT (large, cased), layer 21 |87.14   |64.19   | 56.57   | 52.16    | 41.27   |
|BERT (large, cased), layer 22 |86.87   |64.01   | 55.98   | 51.45    | 40.23   |
|BERT (large, cased), layer 23 |87.02   |63.55   | 55.63   | 50.96    | 40.01   |
|BERT (large, cased), layer 24 |86.69   |63.05   | 55.91   | 50.90    | 40.20   |

![alt-text](https://github.com/DeanSlack/lm_probing/blob/master/figures/bert_large.svg)

### GPT2 (medium)

| Contextualizer              | Leaf    |Parent  | GParent | GGParent | Root     |
|:--------------------------- |:-------:|--------|:-------:|:--------:|:--------:|
|GPT2 (medium), layer 0  | -  |60.22   | 48.69   | 43.19    | 28.55   |
|GPT2 (medium), layer 1  | -  |61.23   | 50.91   | 46.30    | 34.45   |
|GPT2 (medium), layer 2  | -  |61.30   | 51.07   | 46.70    | 34.20   |
|GPT2 (medium), layer 3  | -  |61.34   | 51.08   | 47.01    | 33.88   |
|GPT2 (medium), layer 4  | -  |61.88   | 51.95   | 47.55    | 33.08   |
|GPT2 (medium), layer 5  | -  |62.00   | 52.21   | 47.95    | 33.03   |
|GPT2 (medium), layer 6  | -  |62.22   | 52.70   | 48.41    | 32.78   |
|GPT2 (medium), layer 7  | -  |62.75   | 53.04   | 48.44    | 32.73   |
|GPT2 (medium), layer 8  | -  |62.80   | 53.31   | 48.88    | 33.32   |
|GPT2 (medium), layer 9  | -  |63.04   | 53.89   | 49.46    | 34.36   |
|GPT2 (medium), layer 10 | -  |63.57   | 54.21   | 49.62    | 34.99   |
|GPT2 (medium), layer 11 | -  |63.69   | 54.57   | 50.47    | 35.79   |
|GPT2 (medium), layer 12 | -  |64.41   | 55.26   | 51.21    | 36.45   |
|GPT2 (medium), layer 13 | -  |64.80   | 55.68   | 51.50    | 36.47   |
|GPT2 (medium), layer 14 | -  |65.12   | 56.65   | 51.45    | 36.59   |
|GPT2 (medium), layer 15 | -  |65.37   | 56.69   | 52.03    | 36.61   |
|GPT2 (medium), layer 16 | -  |65.55   | 57.18   | 52.61    | 37.81   |
|GPT2 (medium), layer 17 | -  |65.75   | 57.87   | 53.23    | 38.45   |
|GPT2 (medium), layer 18 | -  |        |         |          | 40.31   |
|GPT2 (medium), layer 19 | -  |65.43   | 57.86   | 53.27    | 40.64   |
|GPT2 (medium), layer 20 | -  |65.38   | 57.80   | 53.35    |         |
|GPT2 (medium), layer 21 | -  |64.19   | 56.57   | 52.16    | 41.27   |
|GPT2 (medium), layer 22 | -  |64.01   | 55.98   | 51.45    | 40.23   |
|GPT2 (medium), layer 23 | -  |63.55   | 55.63   | 50.96    | 40.01   |
|GPT2 (medium), layer 24 | -  |63.05   | 55.91   | 50.90    | 40.20   |
