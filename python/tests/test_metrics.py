import pytest
import numpy as np
import scipy.sparse as sp
from meas.serendipity import *
from meas.diversity import *

SEED = 322


@pytest.fixture
def rank_data():
    # [bool target, score]  # rank in comments
    truth = np.array([
        [0, -10],   # 12
        [0, 0],     # 11
        [1, 2],     # 10
        [0, 3],     # 9
        [0, 4],     # 8
        [0, 5],     # 7
        [1, 10],    # 6
        [0, 11],    # 5
        [0, 20],    # 4
        [1, 50],    # 3
        [0, 70],    # 2
        [0, 100],   # 1
    ])
    np.random.seed(SEED)
    np.random.shuffle(truth)  # should be able to handle rand order

    y_t = truth[:, 0]
    pos_inds = np.where(y_t)[0]
    y_p = truth[:, 1]

    order = np.argsort(-y_p)
    ranks = np.argsort(order) + 1

    # downstream fixtures
    np.testing.assert_array_equal(
        truth,
        np.array([[0, 100],     # 1
                  [1,   2],     # 10
                  [0,  11],     # 5
                  [1,  10],     # 6
                  [0,   3],     # 9
                  [0,  70],     # 2
                  [0,   4],     # 8
                  [0,   0],     # 11
                  [0,  20],     # 4
                  [0,   5],     # 7
                  [1,  50],     # 3
                  [0, -10]])    # 12
    )
    np.testing.assert_array_equal(
        order,
        np.array([0,  5, 10,  8,  2,  3,  9,  6,  4,  1,  7, 11])
    )
    np.testing.assert_array_equal(
        ranks,
        np.array([1, 10,  5,  6,  9,  2,  8, 11,  4,  7,  3, 12])
    )

    return {'y_t': y_t,
            'y_p': y_p,
            'pos_inds': pos_inds,
            'order': order,
            'ranks': ranks,
            }


@pytest.fixture
def multi_user_rank_data():
    """2 users, 3 items"""
    # n_user X n_items score array
    # score_arr = np.array([
    #     [100, 200, 300],
    #     [2000, 1000, 3000],
    # ])
    # This would be calculated from above using argsort
    rank_arr = np.array([
        [3, 2, 1],
        [2, 3, 1],
    ])
    # True targets, (relevance of item)
    truth_arr = np.array([
        [0, 1, 1],
        [1, 1, 0],
    ])

    # These would come from a popularity benchmark or item bias etc
    benchmark_scores = np.array([1, 2, 4])
    benchmark_ranks = np.array([3, 2, 1])

    return {'truth_arr': truth_arr,
            'rank_arr': rank_arr,
            'benchmark_ranks': benchmark_ranks,
            'n_items': truth_arr.shape[1]
            }


@pytest.fixture
def multi_user_rank_data_sp():
    """2 users, 3 items"""
    # n_user X n_items score array
    # score_arr = np.array([
    #     [100, 200, 300],
    #     [2000, 1000, 3000],
    # ])
    # This would be calculated from above using argsort
    rank_arr = np.array([
        [3, 2, 1],
        [2, 3, 1],
    ])
    # True targets, (relevance of item)
    truth_arr = np.array([
        [0, 1, 1],
        [1, 1, 0],
    ])

    rank_sp = sp.csr_matrix(truth_arr * rank_arr)

    # These would come from a popularity benchmark or item bias etc
    benchmark_scores = np.array([1, 2, 4])
    benchmark_ranks = np.array([3, 2, 1])

    return {'rank_sp': rank_sp,
            'benchmark_ranks': benchmark_ranks,
            'n_items': truth_arr.shape[1]
            }


class TestSerendipity(object):

    def test_prob_scalar(self):
        """Test proxy for probability for a single rank"""
        p = prob(5, 10)
        np.testing.assert_equal(p, 5./9)

    def test_prob_arr(self):
        """Test proxy for probability for a single rank"""
        p = prob(np.array([1, 5, 10]), 10)
        np.testing.assert_array_equal(
            p,
            np.array([1., 5./9, 0.])
        )

    def test_serendipity(self, multi_user_rank_data):
        truth_arr = multi_user_rank_data['truth_arr']
        rank_arr = multi_user_rank_data['rank_arr']
        benchmark_ranks = multi_user_rank_data['benchmark_ranks']
        n_items = multi_user_rank_data['n_items']

        s = serendipity(rank_arr, truth_arr, benchmark_ranks)

        np.testing.assert_equal(s, 1./12)


    # def test_mrr(self, rank_data):
    #     pos_inds, order = rank_data['pos_inds'], rank_data['order']
    #     x = recscores.mrr_via_inds_orders(pos_inds, order)
    #     np.testing.assert_almost_equal(
    #         x,
    #         np.mean([1/3, 1/6, 1/10]),
    #     )