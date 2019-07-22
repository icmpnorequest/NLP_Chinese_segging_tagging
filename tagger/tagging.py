# coding=utf8
"""
It is Python3 file to complete pos tagging with segmented word list.
@author: Yantong Lai
@date: 07/18/2019
"""

import re
from tagger.utils import *


class PosTagging():

    def __init__(self):
        """
        term_tag_n: Store all tagged words and their number
        term_list: Store untagged words
        tags_n: Store all tags and their number
        tag_tag_n: Store tag-tag pairs and their number
        states: Store all types of tags

        ##### EXAMPLE #####
        term_tag_n =  {'他/r': 1, '说/v': 1, '，/w': 4, '中/j': 1, '尼/j': 1, '传统/n': 1, '友谊/n': 1, '源远流长/i': 1, '建交/v': 1, '后/f': 1, '两/m': 3, '国/n': 3, '在/p': 1, '政治/n': 1, '、/w': 2, '经济/n': 1, '文化/n': 1, '等/u': 1, '各个/r': 1, '领域/n': 1, '的/u': 2, '合作/vn': 1, '不断/d': 1, '发展/v': 1, '之间/f': 1, '不/d': 1, '存在/v': 1, '任何/r': 1, '问题/n': 1, '关系/n': 2, '堪称/v': 1, '国家/n': 1, '典范/n': 1, '。/w': 1}
        term_list =  ['他', '说', '，', '中', '尼', '传统', '友谊', '源远流长', '，', '建交', '后', '两', '国', '在', '政治', '、', '经济', '、', '文化', '等', '各个', '领域', '的', '合作', '不断', '发展', '，', '两', '国', '之间', '不', '存在', '任何', '问题', '，', '两', '国', '关系', '堪称', '国家', '关系', '的', '典范', '。']
        tags_n = {'r': 3, 'v': 5, 'w': 7, 'j': 2, 'n': 14, 'i': 1, 'f': 2, 'm': 3, 'p': 1, 'u': 3, 'vn': 1, 'd': 2}
        tag_tag_n =  {'Pos_r': 1, 'r_v': 1, 'v_w': 2, 'Pos_j': 1, 'j_j': 1, 'j_n': 1, 'n_n': 3, 'n_i': 1, 'i_w': 1, 'Pos_v': 1, 'v_f': 1, 'f_m': 1, 'm_n': 3, 'n_p': 1, 'p_n': 1, 'n_w': 4, 'Pos_n': 2, 'n_u': 3, 'u_r': 1, 'r_n': 2, 'u_vn': 1, 'vn_d': 1, 'd_v': 2, 'Pos_m': 2, 'n_f': 1, 'f_d': 1, 'v_r': 1, 'n_v': 1, 'v_n': 1, 'u_n': 1}
        states =  ['r', 'v', 'w', 'j', 'n', 'i', 'f', 'm', 'p', 'u', 'vn', 'd']
        """

        self.term_tag_n = {}
        self.tag_tag_n = {}
        self.tags_n = {}
        self.term_list = []
        self.states = []

    def processCorpus(self, path):
        """
        1. It is a function to preprocess the corpus file.
        2. Calculate the init_vec, trans_mat, emit_mat.
        :param path: 199801.txt
        """
        term_list = set()
        with open(file=path, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                # Remove date info
                line = re.sub("\d{8}-\d{2}-\d{3}-\d{3}/m? ", "", line)
                # Split line by "/w"
                sentences = line.split("/w")
                # Add '/w' with Chinese punctuation
                sentences = [term + '/w' for term in sentences[:-1]]

                for sentence in sentences:
                    # Split sentence with terms
                    terms = sentence.split()
                    for i in range(len(terms)):
                        if terms[i] == '':
                            continue
                        try:
                            self.term_tag_n[terms[i]] += 1
                        except KeyError:
                            self.term_tag_n[terms[i]] = 1

                        word_tag = terms[i].split('/')
                        term_list.add(word_tag[0])

                        try:
                            self.tags_n[word_tag[-1]] += 1
                        except KeyError:
                            self.tags_n[word_tag[-1]] = 1

                        if i == 0:
                            tag_tag = 'Pos' + "_" + word_tag[-1]
                        else:
                            tag_tag = terms[i - 1].split('/')[-1] + '_' + word_tag[-1]

                        try:
                            self.tag_tag_n[tag_tag] += 1
                        except KeyError:
                            self.tag_tag_n[tag_tag] = 1

        # Statistics states and term_list
        self.states = list(self.tags_n.keys())
        self.term_list = list(term_list)

        # Calculate init_vec, trans_mat and emit_mat
        self.init_vec = calInitVec(self.tag_tag_n, self.tags_n)
        self.trans_mat = calTransMat(self.tags_n, self.tag_tag_n, self.states)
        self.emit_mat = calEmitMat(self.tags_n, self.term_tag_n, self.term_list, self.states)

    def convertSentence(self, sentence):
        """
        It is a function to convert the sentence to index list.
        :param sentence: segmented sentence
        :return: index list
        """
        res = []
        for word in sentence:
            idx = self.term_list.index(word)
            res.append(idx)
        return res

    def predictTag(self, sentence):
        """
        It is a function to predict tags.
        :param sentence: segmented sentence
        :return: tagged sentence list
        """
        o_seq = self.convertSentence(sentence)
        s_seq = tagViterbi(o_seq, self.trans_mat, self.emit_mat, self.init_vec)

        tagging_list = []
        for i in range(len(o_seq)):
            tag = self.states[s_seq[i]]
            word_tag = self.term_list[o_seq[i]] + '/' + tag
            tagging_list.append(word_tag)
        return tagging_list