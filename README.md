
# **Linear Probing**

Results collected from a simple linear probe layer trained on top of frozen contextualised word representations output from all intermediate layers of two state of the art contextualizers (BERT, ELMo).

## **To-do List**

- Run non-contextual GLoVe word embedding base
- Perform Binary Sentiment Classification
- Re-run experiments with different optimizers
- Implement early stopping with patience
- Check for other papers looking at sentence level sentiment prediction using single word representations
- Change tokenization into function that can be called from dataset, s.t. pass 'moses' tokenize argument and it will call the moses tokenize method
- Changing ELMo model to 5.5b would provide a closer comparison to BERT pretraining data
- Run non-contextual word embedding base
- Check oslo sentiment paper for dataset containing different linguist phenomena


## **Linear Probing Results**

Results for linear probing tasks for all pretrained representations used in the study. All linear models are trained ontop of frozen contextualizers for 60 epochs using the Adam optimizer with a learning rate of 0.0001. Results from the epoch with the best accuracy are used.

## **Sentiment Analysis**
We perform sentence-level sentiment analysis using single word representations contextualised on full sentences. We experiment with both fine-grained (5 classes) and binary, using the Stanford Sentiment Treebank (SST).

### **BERT (base, cased) SST-5 (root sentences)**

12-layer, 768-hidden, 12-heads, 110M parameters.
Trained on cased English text: Wikipedia (~2.5B words) + BookCorpus (~800M words))

| Layer     | Accuracy (%) |
|:---------:|:------------:|
|12         | 39.180       |
|11         | 37.628       |
|10         | 36.734       |
|9          | 35.364       |
|8          | 35.793       |
|7          | 34.464       |
|6          | 33.716       |
|5          | 31.384*       |
|4          | 31.984*       |
|3          | 31.465*       |
|2          | 31.407*       |
|1          | 31.253*       |
|0          | 24.145*       |
*retest

### **ELMo (original) SST-5 (root sentences)**

2-layer (+ embedding layer), 1024-representation, 93.6M parameters.
Trained on cased English text: Google 1 Billion words (~800M tokens)

| Layer     | Accuracy (%) |
|:---------:|:------------:|
|2          | 39.461       |
|1          | 38.206       |
|0          |        |

![Loss Curves ELMo SST5](/figures/elmo_sst5_loss.png){:height="50%" width="50%"}

### **ELMo (5.5B) SST-5 (root sentences)**

2-layer (+ embedding layer), 1024-representation, 93.6M parameters.
Trained on cased English text: 5.5B tokens consisting of Wikipedia (1.9B) and all of the monolingual news crawl data from WMT 2008-2012 (3.6B)

| Layer     | Accuracy (%) |
|:---------:|:------------:|
|2          |        |
|1          |       |
|0          |        |


### **GLoVe (840B.300d) SST-5 (root sentences)**

GLoVe description.

| GLoVe     | Accuracy (%) |
|:---------:|:------------:|
|           |              |

