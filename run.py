# coding = utf8
"""
It is a Python3 file to run Chinese parser.
@author: Hangyu Xia, Yantong Lai
@date: 07/18/2019
"""

# import sys
# from importlib import reload
# reload(sys)
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.externals import joblib
import time
from tagger.seg import HMMSegger
from tagger.tagging import PosTagging


CRF_Model_path = "model/model-1.pkl"


def sentence2feature(sentences):
    """句子形式的语料转化为特征和标签"""
    features, tags = [], []
    for index in range(len(sentences)):
        feature_list, tag_list = [], []
        for i in range(len(sentences[index])):
            feature = {"w0": sentences[index][i][0],
                       "p0": sentences[index][i][1],
                       "w-1": sentences[index][i - 1][0] if i != 0 else "BOS",
                       "w+1": sentences[index][i + 1][0] if i != len(sentences[index]) - 1 else "EOS",
                       "p-1": sentences[index][i - 1][1] if i != 0 else "un",
                       "p+1": sentences[index][i + 1][1] if i != len(sentences[index]) - 1 else "un"}
            feature["w-1:w0"] = feature["w-1"] + feature["w0"]
            feature["w0:w+1"] = feature["w0"] + feature["w+1"]
            feature["p-1:p0"] = feature["p-1"] + feature["p0"]
            feature["p0:p+1"] = feature["p0"] + feature["p+1"]
            feature["p-1:w0"] = feature["p-1"] + feature["w0"]
            feature["w0:p+1"] = feature["w0"] + feature["p+1"]
            feature_list.append(feature)
            tag_list.append(sentences[index][i][-1])
        features.append(feature_list)
        tags.append(tag_list)
    return features, tags


class CRF(object):

    def __init__(self, model_path):
        
        self.model_path = model_path
        self.model = None

    def predict(self, sentences):
        """模型预测"""
        self.load_model()
        features, _ = sentence2feature(sentences)
        return self.model.predict(features)

    def load_model(self, name='model'):
        """加载模型 """
        self.model = joblib.load(self.model_path)

    def save_model(self, name='model'):
        """保存模型"""
        joblib.dump(self.model, self.model_path)


def sent_packer(sent):  # 将一句中文句子sen进行分词、词性标注，并处理为上面函数可以处理的数据格式(三维list)
    '''
    :param sent: 世界第八大奇迹出现
    :return: A = [[['世界', 'n', 'n'],
            ['第', 'm', 'm'],
            ['八', 'm', 'm'],
            ['大', 'a', 'a'],
            ['奇迹', 'n', 'n'],
            ['出现', 'v', 'v']
            ]]
    '''
    # 世界第八大奇迹出现

    '''1. Segment the sentence'''
    print("1. Segging ...")
    # Create an instance
    segger = HMMSegger()
    # Load data
    segger.load_data("data/train_seg_corpus.txt_utf8")
    # Train
    segger.train()
    seg_sent = segger.cut(sent)
    print("seg_sent = ", seg_sent)
    # seg_sent =  ['世界', '第八', '大', '奇迹', '出现']

    '''2. Tag the segmented sentence'''
    print("2. Tagging ...")
    # Create an instance
    tagger = PosTagging()
    # Load data
    tagger.processCorpus("data/199801.txt")
    seg_list = tagger.predictTag(seg_sent)
    # seg_list = ['世界/n', '第八/m', '大/a', '奇迹/n', '出现/v']

    '''3. Process the seg_list'''
    A = []
    M = []
    for item in seg_list:
        item_split = item.split("/")
        # item_split[0]: word
        # item_split[1]: tag
        temp = [item_split[0], item_split[1], item_split[1]]
        M.append(temp)
    A.append(M)
    return A


if __name__ == "__main__":

    crf_parser = CRF(model_path=CRF_Model_path)
    crf_parser.load_model()      # 加载已训练好的"model.pkl"文件

    # sen = [[['世界', 'n', 'n', '1_n'],
    #         ['第', 'm', 'm', '1_a'],
    #         ['八', 'm', 'm', '-1_m'],
    #         ['大', 'a', 'a', '1_n'],
    #         ['奇迹', 'n', 'n', '1_v'],
    #         ['出现', 'v', 'v', '0_Root']
    #         ]]

    # sen = [[['世界', 'n', 'n'],
    #         ['第', 'm', 'm'],
    #         ['八', 'm', 'm'],
    #         ['大', 'a', 'a'],
    #         ['奇迹', 'n', 'n'],
    #         ['出现', 'v', 'v']
    #         ]]

    # sent = "我吃了面包"
    # sent = "世界第八大奇迹出现"
    sent = input("请输入一句不带标点符号的中文句子：\n")
    sen = sent_packer(sent)
    print('tag_sent = ', sen)

    print("3. Parsing ...")
    print("依存句法分析结果为：", crf_parser.predict(sen)[0])    # 输入前三列,输出最后一列
