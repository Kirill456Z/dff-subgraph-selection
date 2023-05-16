import unittest
import pickle
from utils import predict_dial, calc_cluster_pred_acc
from dataset.dataset import Utterance, Dialogue
import numpy as np

class TestDialPredict(unittest.TestCase):
    def setUp(self) -> None:
        self.graphs = [ "graph_2_60_sbert", "graph_2_120_sbert", "graph_2_180_sbert",
                        "selected_sbert_10", "selected_sbert_14", "selected_sbert_20" ]

        utts = [Utterance("i need a place to dine in the center thats expensive", "USER", 0), 
                Utterance("I have several options for you; do you prefer African, Asian, or British food?", "SYSTEM", 1),
                Utterance("Any sort of food would be fine, as long as it is a bit expensive. Could I get the phone number for your recommendation?", "USER", 2),
                Utterance("There is an Afrian place named Bedouin in the centre. How does that sound?", "SYSTEM", 3), 
                Utterance("Sounds good, could I get that phone number? Also, could you recommend me an expensive hotel?", "USER", 4),
                Utterance("Bedouin's phone is 01223367660. As far as hotels go, I recommend the University Arms Hotel in the center of town.", "SYSTEM", 5),
                Utterance("Yes. Can you book it for me?" , "USER", 6),
                Utterance("Sure, when would you like that reservation?", "SYSTEM", 7),
                Utterance("i want to book it for 2 people and 2 nights starting from saturday.", "USER", 8),
                Utterance("Your booking was successful. Your reference number is FRGZWQL2 . May I help you further?", "SYSTEM", 9),
                Utterance("That is all I need to know. Thanks, good bye.", "USER", 10),
                Utterance("Thank you so much for Cambridge TownInfo centre. Have a great day!", "SYSTEM", 11)]

        self.test_dial = Dialogue(utts, "TestDial0", services = "")
    
    def test_cluster_pred_acc(self):
        expected_acc = 0.15
        for cur_graph_name in self.graphs:
            with open(f'dialogue_sim/graphs/{cur_graph_name}.pkl', 'rb') as file:
                graph = pickle.load(file)
            dialogue_data = predict_dial(self.test_dial, graph)
            acc = calc_cluster_pred_acc(dialogue_data)
            self.assertGreater(acc, expected_acc, f"cluster prediction accuracy for graph {cur_graph_name} is too low. Got {acc}, expected at least {expected_acc}")

    def test_predictions_match_with_precomputed(self):
        EMB_PATH = 'embeddings/multiwoz/sentence_bert_{split}_embeddings.npy'
        train_emb = np.load(EMB_PATH.format(split = 'train'))

        for cur_graph_name in self.graphs:
            with open(f'dialogue_sim/graphs/{cur_graph_name}.pkl', 'rb') as file:
                graph = pickle.load(file)
                precomputed = DialogueData(train[0], train, train_emb, graph)
                predicted = predict_dial(self.test_dial)
                pred_l = len(predicted.second_stage_clusters)
                prec_l = len(precomputed.second_stage_clusters)
                self.assertEqual(pred_l, prec_l, f"prediction resulted in {pred_l} clusters, while should be {prec_l}")
                for i in range(len(predicted.second_stage_clusters)):
                    pred_c = predicted.second_stage_clusters[i]
                    prec_c = precomputed.second_stage_clusters[i]
                    self.assertEqual(pred_c, prec_c, f"predicted cluster no {pred_c} in position {i}, while should be {prec_c}")


if __name__ == '__main__':
    unittest.main()