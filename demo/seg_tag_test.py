# coding = utf8
"""
It is a Chinese seg + pos tagging tool demo.
@author: Yantong Lai
@date: 07/18/2019
"""

from tagger.seg import HMMSegger
from tagger.tagging import PosTagging

if __name__ == '__main__':

    test = "世界第八大奇迹出现"
    print("Original string = ", test)

    '''1. Segment the sentence'''
    # Create an instance
    segger = HMMSegger()
    # Load data
    segger.load_data("../data/train_seg_corpus.txt_utf8")
    # Train
    segger.train()
    seg_sent = segger.cut(test)
    print('Seg res = ', seg_sent)

    '''2. Tag the segmented sentence'''
    # Create an instance
    tagger = PosTagging()
    # Load data
    tagger.processCorpus("../data/199801.txt")
    res = tagger.predictTag(seg_sent)
    print('Tagging res = ',res)