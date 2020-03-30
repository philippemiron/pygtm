import numpy as np


# different tool functions used across the classes
def ismember(a, b):
    """
    Re-implementation of ismember() from Matlab but return only the second arg which is the index
    https://stackoverflow.com/questions/15864082/python-equivalent-of-matlabs-ismember-function
    Args:
        a: first list
        b: second list to compare

    Returns:
        list size of a: value is the indices in the list b (-1 if not present)
        ex: ismember([1,2,4], [1,2,3,5]) = [0, 1, -1]
    """
    bind = {}
    for i, elt in enumerate(b):
        if elt not in bind:
            bind[elt] = i
    return np.array([bind.get(itm, -1) for itm in a])  # -1 if not in b


def filter_vector(vector, keep):
    """
    Remove element from vector (or vectors is a list) according to a index list or boolean vector
    vectors: Numpy array or list of Numpy array
    Args:
        vector: array or list of arrays to filter
        keep: list of index to keep or a boolean array with the same size as vector

    Returns:
        outlist: filtered array or list of arrays
    """
    if type(vector) != list:
        return vector[keep]
    else:
        outlist = []
        for vec in vector:
            outlist.append(vec[keep])
        return outlist
