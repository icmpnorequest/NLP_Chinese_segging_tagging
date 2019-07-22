# coding=utf8
"""
@Code Description: It is a Python3 file to complete Chinese segmentation tool.
@Reference: jieba
@author: Yantong Lai
@date: 07/18/2019
"""

from tagger.extra import seg_stop_words
from tagger.hmm import HMM

"""
E: The last character of the word
B: The first character of the word
M: The middle character of the word
S: The word with a single character
"""
STATES = {'B', 'M', 'E', 'S'}


def getTags(src):
    """
    It is a function to get sentence tags with {'B', 'M', 'E', 'S'}.
    :param src:
    :return: tags list
    """
    tags = []
    if len(src) == 1:
        tags = ['S']
    elif len(src) == 2:
        tags = ['B', 'E']
    else:
        m_num = len(src) - 2
        tags.append('B')
        tags.extend(['M'] * m_num)
        tags.append('S')
    return tags

def cutSent(src, tags):
    """
    It is a function to cut sentence.
    :param src
    :param tags
    :return:
    """
    word_list = []
    start = -1
    started = False

    if len(tags) != len(src):
        return None

    if tags[-1] not in {'S', 'E'}:
        if tags[-2] in {'S', 'E'}:
            # for tags: r".*(S|E)(B|M)"
            tags[-1] = 'S'
        else:
            # for tags: r".*(B|M)(B|M)"
            tags[-1] = 'E'
    for i in range(len(tags)):
        if tags[i] == 'S':
            if started:
                started = False
                # for tags: "BM*S"
                word_list.append(src[start:i])
            word_list.append(src[i])
        elif tags[i] == 'B':
            if started:
                # for tags: "BM*B"
                word_list.append(src[start:i])
            start = i
            started = True
        elif tags[i] == 'E':
            started = False
            word = src[start:i+1]
            word_list.append(word)
        elif tags[i] == 'M':
            continue
    return word_list


class HMMSegger(HMM):

    def __init__(self, *args, **kwargs):
        super(HMMSegger, self).__init__(*args, **kwargs)
        self.states = STATES
        self.data = None

    def load_data(self, filename):
        """
        It is function to load data
        :param filename: corpus file
        """
        self.data = open(filename, 'r', encoding="utf-8")

    def train(self):
        """
        It is a function to preprocess, get observes and states.
        Count observes and states.
        """
        if not self.inited:
            self.setup()

        # train
        for line in self.data:
            # preprocessing
            line = line.strip()
            if not line:
                continue

            # get observes
            observes = []
            for i in range(len(line)):
                if line[i] == " ":
                    continue
                observes.append(line[i])

            # get states
            # split line by space
            words = line.split(" ")
            states = []
            for word in words:
                if word in seg_stop_words:
                    continue
                states.extend(getTags(word))
            # resume train
            self.do_train(observes, states)

    def cut(self, sentence):
        """
        It is a function complete sentence segmentation.
        :param sentence:
        :return: segmented sentence
        """
        try:
            # calculate probability
            tags = self.viterbi(sentence, self.init_vec, self.trans_mat, self.emit_mat)
            return cutSent(sentence, tags)
        except:
            return sentence