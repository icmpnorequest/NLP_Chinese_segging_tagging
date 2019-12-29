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



## 3. Result

![Image text](https://github.com/icmpnorequest/NLP_Chinese_segging_tagging/blob/master/image/test_res.png)


## 4. License

MIT License

Copyright (c) 2019 icmpnorequest

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.



