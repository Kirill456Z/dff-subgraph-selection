import unittest
import pickle
from dialogue_sim.sim_metrics import MetricsEval

class TestMetrics(unittest.TestCase):
    def setUp(self) -> None:
        with open("dialogue_sim/data/test_data.pkl", "rb") as file:
            self.test_dials = pickle.load(file)
        self.metrics_evaluator = MetricsEval()
        self.failures = []

    def test_self_similarity(self):
        """
        Test that similarity metrics are high on same dialogue
        """
        test_dial = self.test_dials[0]
        metrics = self.metrics_evaluator.get_metrics(test_dial, test_dial)

        expected_values = {
            'mean_first_stage_emb' : 1,
            'node_intersection' : len(test_dial.second_stage_clusters),
            'dice' : 1,
            'distr_intersect' : 1,
            'JS_distance' : 0,
            'distr_cosine' : 1,
            'mean_lm_embeddings' : 1,
        }

        expected_gt = {
            'FFN' : 0.5,
            'GRU' : 0.5,
            'LSTM' : 0.5
        }

        for metric_name, value in expected_values.items():
            self.assertEqual(value, metrics[metric_name], f"metric {metric_name} should be {value} on same dialogue, but got {metrics[metric_name]}")
        
        for metric_name, value in expected_gt.items():
            self.assertGreater(metrics[metric_name], value, f"metric {metric_name} should be greater than {value} but got {metrics[metric_name]}")

    
    def test_low_similarity(self):
        """
        Test that similarity metrics are low on dialogues from different domains
        """
        test_dial1 = self.test_dials[0]
        test_dial2 = self.test_dials[1]

        # Assert that dialogues are from different domains
        domain_intersect = set.intersection(set(test_dial1.services), set(test_dial2.services))
        self.assertEqual(len(domain_intersect), 0)

        metrics = self.metrics_evaluator.get_metrics(test_dial1, test_dial2)

        expected_lt = {
            'mean_first_stage_emb' : 0.5,
            'node_intersection' : 3,
            'dice' : 0.5,
            'distr_intersect' : 0.5,
            'JS_distance' : 0.55,
            'distr_cosine' : 0.5,
            'FFN' : 0.5,
            'GRU' : 0.5,
            'LSTM' : 0.5,
        }

        for metric_name, value in expected_lt.items():
            self.assertLess(metrics[metric_name], value, f"metric {metric_name} should be less than {value}, but got {metrics[metric_name]}")


if __name__ == '__main__':
    unittest.main()