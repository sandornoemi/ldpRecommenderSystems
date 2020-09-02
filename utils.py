import numpy as np
import pandas as pd
from copy import deepcopy


def special_sort(a, order='ascending'):
    n = len(a)

    if order == 'ascending':
        for i in range(1, n):
            j = deepcopy(i)

            while j > 0 and a[j][1] < a[j - 1][1]:
                temp = a[j - 1]
                a[j - 1] = a[j]
                a[j] = temp

                j = j - 1

    elif order == 'descending':
        for i in range(1, n):
            j = deepcopy(i)

            while j > 0 and a[j][1] > a[j - 1][1]:
                temp = a[j - 1]
                a[j - 1] = a[j]
                a[j] = temp

                j = j - 1
    return a


def dissimilarity(arr1, arr2, weighted):
    n = arr1.shape[0]
    s = 0
    if weighted:
        for i in range(0, n):
            diff = abs(arr1[i] - arr2[i])
            s = s + (diff * (n - i) / n)
    else:
        for i in range(0, n):
            diff = abs(arr1[i] - arr2[i])
            s = s + diff
    return s
