# coding = utf8
"""
It is a Chinese pos tagging tool demo.
@author: Yantong Lai
@date: 07/18/2019
"""

from tagger.tagging import PosTagging

if __name__ == '__main__':

    # Create an instance
    pt = PosTagging()
    # Load data
    pt.processCorpus("../data/199801.txt")
    res = pt.predictTag(['世界', '第八', '大', '奇迹', '出现'])
    print('res = ', res)