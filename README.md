
# **Linear Probing**

Results collected from a simple linear probe layer trained on top of frozen contextualised word representations output from all intermediate layers of two state of the art contextualizers (BERT, ELMo). Much work has been done to analyse the internal representations and geometry of pretrained language models, in terms of syntactic and semantic subspaces. However, little has been done to determine how architecture-specific internal representations are mapped to a single contextual word embedding.

## **To-do List**

- Tabulate POS results
- Perform Binary Sentiment Classification
- Implement early stopping with patience
- Experiment will full treebank phrases for training
- Run all pretrained representations on non-linear classifier to determine the performance being constrained by using linear models.
- Check for other papers looking at sentence level sentiment prediction using single word representations
- Change tokenization into function that can be called from dataset, s.t. pass 'moses' tokenize argument and it will call the moses tokenize method
- Changing ELMo model to 5.5b would provide a closer comparison to BERT pretraining data
- Check oslo sentiment paper for dataset containing different linguist phenomena
- Qualitative assessment for correct classifications which override/negate the ground sentiment of the word with no context.


## **Linear Probing Results**

Results for linear probing tasks for all pretrained representations used in the study. All linear models are trained ontop of frozen contextualizers for 60 epochs using the Adam optimizer with a learning rate of 0.0001. Results from the epoch with the best accuracy are used.

## **Sentiment Analysis**

We perform sentence-level sentiment analysis using single word representations contextualised on full sentences. We experiment with both fine-grained (5 classes) and binary, using the Stanford Sentiment Treebank (SST).

### **BERT (base, cased) SST-5 (root sentences)**

12-layer, 768-hidden, 12-heads, 110M parameters.
Trained on cased English text: Wikipedia (~2.5B words) + BookCorpus (~800M words))

| Layer     | SST-5    | SST-2                |
|:---------:|:---------|:--------------------:|
|12         | 39.180   |
|11         | 37.628   |
|10         | 36.734   |
|9          | 35.364   |
|8          | 35.793   |
|7          | 34.464   |
|6          | 33.716   |
|5          | 31.384*  |
|4          | 31.984*  |
|3          | 31.465*  |
|2          | 31.407*  |
|1          | 31.253*  |
|0          | 24.145*  |

*retest

### **ELMo (original) SST-5 (root sentences)**

2-layer (+ non-contextual embedding layer), 1024-representation, 93.6M parameters.
Trained on cased English text: Google 1 Billion words (~800M tokens)

| Layer     | SST-5                | SST-2                |
|:---------:|:--------------------:|:--------------------:|
|2          | 39.461               |                      |
|1          | 38.206               |                      |
|0          | 29.097               |                      |

### **ELMo (5.5B) SST-5 (root sentences)**

2-layer (+ non-contextual embedding layer), 1024-representation, 93.6M parameters.
Trained on cased English text: 5.5B tokens consisting of Wikipedia (1.9B) and all of the monolingual news crawl data from WMT 2008-2012 (3.6B)

| Layer     | SST-5     | SST-2                |
|:---------:|:---------:|:--------------------:|
|2          | 40.151    | 77.013
|1          | 38.701    |
|0          | 29.052    |

### **ELMo (4-layer) SST-5 (root sentences)**

4-layer (+ non-contextual embedding layer), 1024-representation, 93.6M parameters.
Trained on cased English text: Google 1 Billion words (~800M tokens)

| Layer     | SST-5     | SST-2                |
|:---------:|:---------:|:--------------------:|
|4          |
|3          |
|2          |
|1          |
|0          |

### **GLoVe (840B.300d) SST-5 (root sentences)**

GLoVe description.

|           | SST-5                | SST-2                |
|:---------:|:--------------------:|:--------------------:|
| GLoVe     | 28.808               | 57.721               |
