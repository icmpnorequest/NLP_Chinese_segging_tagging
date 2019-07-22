# coding=utf8
"""
It is a Python3 file to implement some utils function
@author: Yantong Lai
@date: 07/18/2019
"""

import numpy as np

def calInitVec(tag_tag_n, tags_n):
    """
    It is a function to calculate init vector.
    :param tag_tag_n: Tag-tag type and its number
    :param tags_n: Tag type and its number
    :return: init vec
    """
    # Calculate sum of tag_tag_n
    sum_tag = sum(list(tag_tag_n.values()))
    init_vec = [tags_n[value] / sum_tag for value in tags_n]
    return init_vec

def calTransMat(states_n, state_state_n, states):
    """
    It is a function to calculate trans mat.
    :param states_n: tagging type and its number, a dict
    :param state_state_n: state-state pair type and its number, a dict
    :param states: tagging type, a list
    :return: matrix
    """
    # Create a matrix object
    trans_mat = np.zeros((len(states_n), len(states_n)), dtype=float)

    for state1 in range(len(states_n)):
        for state2 in range(len(states_n)):
            state_pair = states[state1] + '_' + states[state2]
            try:
                # Calculate p(state2 | state1)
                trans_mat[state1, state2] = state_state_n[state_pair] / (states_n[states[state1]] + 1)
            except KeyError:
                trans_mat[state1, state2] = 0.0
    return trans_mat

def calEmitMat(states_n, o_state_n, o_sequence, states):
    """
    It is a function to calculate emit matrix.
    :param states_n: states type and its number
    :param o_state_n: tagged word number
    :param o_sequence: word
    :param states: types of all tags
    :return:
    """
    emit_mat = np.zeros((len(states), len(o_sequence)), dtype=float)

    for i in range(len(states)):
        for j in range(len(o_sequence)):
            s = o_sequence[j] + '/' + states[i]
            tag_i = states[i]
            try:
                emit_mat[i, j] = o_state_n[s] / states_n[tag_i]
            except KeyError:
                emit_mat[i, j] = 0
    return emit_mat

def tagViterbi(o_sequence, A, B, pi):
    """
    It is a function to complete tagging Viterbi decode.
    :param o_sequence: observe
    :param A: trans_mat
    :param B: emit_mat
    :param pi: init_vec
    :return:
    """
    len_status = len(pi)
    # Build a dynamic table
    status_record = {i: [[0, 0] for j in range(len(o_sequence))] for i in range(len_status)}

    for i in range(len(pi)):
        status_record[i][0][0] = pi[i] * B[i, o_sequence[0]]
        status_record[i][0][1] = 0

    for t in range(1, len(o_sequence)):
        for i in range(len_status):
            max = [-1, 0]
            for j in range(len_status):
                # Calculate probability
                tmp_prob = status_record[j][t - 1][0] * A[j, i]
                if tmp_prob > max[0]:
                    max[0] = tmp_prob
                    max[1] = j
            # Update table
            status_record[i][t][0] = max[0] * B[i, o_sequence[t]]
            status_record[i][t][1] = max[1]

    # Search best path
    max = 0
    max_idx = 0
    t = len(o_sequence) - 1
    for i in range(len_status):
        if max < status_record[i][t][0]:
            max = status_record[i][t][0]
            max_idx = i

    # Stack
    state_sequence = []
    state_sequence.append(max_idx)
    while (t > 0):
        max_idx = status_record[max_idx][t][1]
        state_sequence.append(max_idx)
        t -= 1
    state_sequence.reverse()
    return state_sequence