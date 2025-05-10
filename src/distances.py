
import numpy as np


def manhattan_distance(pos1, pos2):
    return np.abs(pos1[0] - pos2[0]) + np.abs(pos1[1] - pos2[1])


def p_distance(pos1, pos2, p=2):
    """
    returns the p-distance between pos1 and pos2,
    where p denotes the exponent, e.g. 2-distance yields the l2-norm
    """
    return np.sum([np.abs(x1 - x2) ** p for (x1, x2) in zip(pos1, pos2)]) ** 1.0/p

