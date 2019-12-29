#!/usr/bin/env python
"""
@author: Zezhong Han
"""

import sys
import json
from collections import defaultdict
from six.moves import xrange
"""
概率上下文无关文法（Probabilistic Context-Free Grammar Parser）
本程序基于PCFG输出一个句子的句法解析树
"""

def read_counts(counts_file):
    """
    从文件中读取数据返回一个迭代器
    """
    try:
        fi = open(counts_file, 'r')
    except IOError:
        sys.stderr.write('ERROR: Cannot open %s.\n' % counts_file)
        sys.exit(1)

    for line in fi:
        fields = line.strip().split(' ')
        yield fields


class PCFGParser():
    """
    跟酒文件中存储的每一个非终结符，二元规则和一元规则
    计算每一条规则的概率
    通过Coke-Younger-Kasami (CYK)算法从输入流中解析输入的句子
    以JSON的格式输出一个解析树
    """
    def __init__(self):
        self.nonterminal_counts = defaultdict(int)
        self.binary_rule_counts = defaultdict(int)
        self.unary_rule_counts = defaultdict(int)

    def train(self, counts_file):
        """
        从统计文件中读取每一种规则的计数，然后又存储下来
        我们将规则分成三种类型
        nonterminal, binary rule 和 unary rule.
        """
        for l in read_counts(counts_file):
            n, count_type, args = int(l[0]), l[1], l[2:]
            if count_type == 'NONTERMINAL':
                self.nonterminal_counts[args[0]] = n
            elif count_type == 'BINARYRULE':
                self.binary_rule_counts[tuple(args)] = n
            else: # UNARYRULE counts
                self.unary_rule_counts[tuple(args)] = n

    def q(self, x, y1, y2):
        """
        返回某一条二元规则的概率
        """
        return float(self.binary_rule_counts[x, y1, y2]) / self.nonterminal_counts[x]

    def q_unary(self, x, w):
        """
        返回某条一元规则的概率
        """
        return float(self.unary_rule_counts[x, w]) / self.nonterminal_counts[x]

    def parse(self, sentences):
        """
        通过CKY算法解析句子
        将解析树以JSON格式输出
        """
        for s in sentences:
            s = s.strip()
            if s:
                print(json.dumps(self.CKY(s.split(' '))))

    def CKY(self, x):
        """
        CKY 算法的实现.
        返回一个句子的解析树，本算法假定语法符合Chomsky文法
        """
        n = len(x) # 句子 x的长度
        pi = defaultdict(float) # DP表 pi
        bp = {} # 后退指针
        N = self.nonterminal_counts.keys() # 非终结符集合

        # 基本条件下
        for i in range(n):
            if sum([self.unary_rule_counts[X, x[i]] for X in N]) < 5: # 如果 x[i] 是不频繁的词（小于五次）
                w = '_RARE_' # 用 _RARE_ 代替真实的词
            else:
                w = x[i] 
            for X in N:
                pi[i, i, X] = self.q_unary(X, w) # 如果 X -> x[i] 不在规则列表中, 赋值为 0
        
        # 递归情况
        for l in xrange(1, n): 
            for i in xrange(n-l):
                j = i + l
                for X in N:
                    max_score = 0
                    args = None
                    for R in self.binary_rule_counts.keys(): # 只搜索非零概率的规则
                        if R[0] == X: # 考虑以X为开头的情况
                            Y, Z = R[1:]
                            for s in xrange(i, j):
                                if pi[i, s, Y] and pi[s + 1, j, Z]:
                                    score = self.q(X, Y, Z) * pi[i, s, Y] * pi[s + 1, j, Z]
                                    if max_score < score:
                                        max_score = score
                                        args = Y, Z, s
                    if max_score: # 更新DP表以及后退指针
                        pi[i, j, X] = max_score
                        bp[i, j, X] = args

    
        if pi[0, n-1, 'S']:
            return self.recover_tree(x, bp, 0, n-1, 'S')
        else: # 如果解析树没有以符号‘S’作为开始
            max_score = 0
            args = None
            for X in N:
                if max_score < pi[0, n-1, X]:
                    max_score = pi[0, n-1, X]
                    args = 0, n-1, X
            return self.recover_tree(x, bp, *args)

    def recover_tree(self, x, bp, i, j, X):
        """
        返回解析树
        """
        if i == j:
            return [X, x[i]]
        else:
            Y, Z, s = bp[i, j, X]
            return [X, self.recover_tree(x, bp, i, s, Y), 
                       self.recover_tree(x, bp, s+1, j, Z)]


def usage():
    print('用法1: python pcfg.py [counts_file] < [input_file] 从统计文件中读取规则及频数训练PCFG解析器，然后解析输入文件中的句子\n')
    print('用法2：echo "your sentences" | python pcfg.py [counts_file]\n ')
    print('例如： echo "the man saw the dog with a telescope" | python pcfg.py English.rules')
if __name__ == '__main__':
    if len(sys.argv) != 2: # 如果参数个数不是两个，那么提示用法
        usage()
        sys.exit(2)

    parser = PCFGParser() # 初始化一个 PCFG 解析器
    parser.train(sys.argv[1]) # 通过文件来计算规则及概率
    parser.parse(sys.stdin) # 解析输入流中的句子

