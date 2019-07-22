# NLP_Chinese_segging_tagging

## 1. Project Description
The project aims to build a Chinese segging and POS tagging tool with HMM + Viterbi.

## 2. Code Description
### 2.1 Corpus

/data

- train_seg_corpus.txt_utf8: For Chinese seggmentation
- 199801.txt: For Chinese POS tagging

### 2.2 Main Function

/tagger

- extra.py: Save segged stop words
- hmm.py: It is a Python3 file to complete HMM algorithm (Mainly for Segging)
- seg.py: Segging
- tagging.py: Chinese POS tagging
- utils.py: Tagging utils

### 2.3 Demo

/demo

- seg_test.py: Segging demo
- tag_test.py: Tagging demo
- seg_tag_test.py: Segging + Tagging demo
