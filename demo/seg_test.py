# coding = utf8
"""
It is Chinese segmentation tool demo
@author: Yantong Lai
@date: 07/18/2019
"""

from tagger.seg import HMMSegger


if __name__ == '__main__':

    # Create an instance
    segger = HMMSegger()
    # Load data
    segger.load_data("../data/train_seg_corpus.txt_utf8")
    # Train
    segger.train()
    test = "世界第八大奇迹出现"
    res = segger.cut(test)
    print(res)