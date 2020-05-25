from lime.lime_base import LimeBase
from lime.explanation import Explanation, DomainMapper
from functools import partial
from sklearn.utils import check_random_state
import scipy as sp
import sklearn


class GraphDomainMapper(DomainMapper):
    """Maps nodes indices to vertices names"""

    def __init__(self, neighbors_lst):
        """Initializer.
        Args:
            neighbors_lst: list of neighbors as strings.
        """
        super().__init__()

        # sort neighbors in lexicographical order
        self._neighbors_lst = sorted(neighbors_lst)

    def map_exp_ids(self, exp, **kwargs):
        """Maps indices to nodes names.
        Args:
            exp: list of tuples [(idx, weight), (idx,weight)]
        Returns:
            list of tuples (node, weight)
        """

        exp = [(self._neighbors_lst[x[0]], x[1]) for x in exp]
        return exp

