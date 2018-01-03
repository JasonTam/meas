from sklearn.metrics.pairwise import pairwise_distances


def prob(rank, n):
    """Proxy for the probability of an item being recommended for a user
        given its rank
    
    Args:
        rank: rank of item(s)
            can be a single rank, or an array of ranks
        n: total number of items

    Returns:
        p: probability proxy
            Will be a single value or an array depending on the type of `rank`
            
    TODO:
        safeguard against n<=1

    """
    p = (n - rank) / (n - 1.)
    return p


def serendipity(rank_arr, truth_arr,
                benchmark_ranks,
                n_items=None):
    """Checks the difference in ranking compared to a
        standard benchmark ranking
        
    Examples:
        Compare fancy recommender system to a popular-item benchmark
    
    Args:
        rank_arr: ranks of all user*item
        truth_arr: true relevancy for all user*item
        benchmark_ranks: benchmark ranking for all items
        n_items: number of items
            if `None`, will infer from the shape of `benchmark_ranks`

    Returns:
        average serendipity score
        
    TODO:
        might want to pass in sparse matrices to combine rank & truth array
        ie) have a rank entry only if it's a relevant item

    """

    if n_items is None:
        n_items = len(benchmark_ranks)

    prob_arr = prob(rank_arr, n_items)
    benchmark_probs = prob(benchmark_ranks, n_items)
    s = truth_arr*(prob_arr - benchmark_probs).clip(min=0)

    # Mean over all users and all items
    # TODO: maybe only mean over relevant items
    return s.mean()


def historical_similarity(x_cur, x_hist):
    """ Checks how similar the current set is to a historical set
    
    Returns:

    """
    # Note: can also maybe just pdist(xcur.mean, x_hist.mean)

    return pairwise_distances(x_cur, x_hist).mean()
