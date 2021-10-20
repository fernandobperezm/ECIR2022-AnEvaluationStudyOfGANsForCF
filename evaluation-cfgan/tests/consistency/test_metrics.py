import numpy as np

from conferences.cikm.cfgan.original_source_code.evaluation import computeTopNAccuracy as cfgan_compute_top_n_accuracy
from recsys_framework.Evaluation.metrics import ndcg, precision, recall


class TestConsistencyOfMetrics:
    TEST_RECOMMENDATIONS = [
        [1, 2, 3, 4, 5, 6, 7],
        [5, 6, 8, 9, 4, 3, 1],
        [0, 8, 6, 9, 4, 2, 5],
    ]
    TEST_GROUND_TRUTH = [
        [1, 7, 2, 10, 12, 13],
        [5, 7, 2],
        [9, 10, 12, 15, 17]
    ]
    TEST_CUT_OFF = 5
    TEST_NUM_USERS = len(TEST_RECOMMENDATIONS)

    def test_ndcg_is_equal(self):
        # Arrange

        # Act
        _, _, cfgan_ndcg, _ = cfgan_compute_top_n_accuracy(
            GroundTruth=self.TEST_GROUND_TRUTH,
            predictedIndices=self.TEST_RECOMMENDATIONS,
            topN=[self.TEST_CUT_OFF],
        )

        our_ndcg = 0.0
        for user_index in range(self.TEST_NUM_USERS):
            ranked_list = np.array(self.TEST_RECOMMENDATIONS[user_index])
            pos_items = np.array(self.TEST_GROUND_TRUTH[user_index])
            relevance = np.ones_like(self.TEST_GROUND_TRUTH[user_index])

            our_ndcg += ndcg(
                ranked_list=ranked_list,
                pos_items=pos_items,
                relevance=relevance,
                at=self.TEST_CUT_OFF
            )
        our_ndcg /= self.TEST_NUM_USERS

        # Assert
        assert np.equal(cfgan_ndcg[0], our_ndcg)

    def test_precision_is_equal(self):
        # Arrange

        # Act
        cfgan_precision, _, _, _ = cfgan_compute_top_n_accuracy(
            GroundTruth=self.TEST_GROUND_TRUTH,
            predictedIndices=self.TEST_RECOMMENDATIONS,
            topN=[self.TEST_CUT_OFF],
        )

        our_precision = 0.0
        for user_index in range(self.TEST_NUM_USERS):
            is_relevant = np.in1d(
                self.TEST_RECOMMENDATIONS[user_index][:self.TEST_CUT_OFF],
                self.TEST_GROUND_TRUTH[user_index],
                assume_unique=True
            )

            our_precision += precision(
                is_relevant=is_relevant,
            )
        our_precision /= self.TEST_NUM_USERS

        # Assert
        assert np.equal(cfgan_precision[0], our_precision)

    def test_recall_is_equal(self):
        # Arrange

        # Act
        _, cfgan_recall, _, _ = cfgan_compute_top_n_accuracy(
            GroundTruth=self.TEST_GROUND_TRUTH,
            predictedIndices=self.TEST_RECOMMENDATIONS,
            topN=[self.TEST_CUT_OFF],
        )

        our_recall = 0.0
        for user_index in range(self.TEST_NUM_USERS):
            is_relevant = np.in1d(
                self.TEST_RECOMMENDATIONS[user_index][:self.TEST_CUT_OFF],
                self.TEST_GROUND_TRUTH[user_index],
                assume_unique=True
            )

            our_recall += recall(
                is_relevant=is_relevant,
                pos_items=np.array(self.TEST_GROUND_TRUTH[user_index])
            )
        our_recall /= self.TEST_NUM_USERS

        # Assert
        assert np.equal(cfgan_recall[0], our_recall)
