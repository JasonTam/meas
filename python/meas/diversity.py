import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


def intra_list_distance(x, **kwargs):
    """Calculates the average distance between all distinct pairs
    
    Args:
        x: array of shape [n_record, n_features]
        **kwargs: arguments to pass onto 
            `sklearn.metrics.pairwise.pairwise_distances`

    Returns:
        avg_dist: intra-list distance
    
    References:
        .. [1] Cai-Nicolas Zeigler, et al 2005, Improving Recommendation Lists 
        Through Topic Diversification. 
        http://files.grouplens.org/papers/ziegler-www05.pdf

    """
    dist_mat = pairwise_distances(x, **kwargs)
    triu_inds = np.triu_indices_from(dist_mat, k=1)
    avg_dist = dist_mat[triu_inds].mean()
    return avg_dist


def intra_list_similarity(x, **kwargs):
    """See `intra_list_distance`"""
    return 1. - intra_list_similarity(x, **kwargs)


def content_coverage():
    pass


def category_diversity():
    pass
