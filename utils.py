from embedders.sentence_embedder import SentenceEmbedder
from dialogue_sim.dialogue_data import DialogueData
import numpy as np

def predict_dial(dial, graph, emb = None):
    if emb is None:
        emb = SentenceEmbedder()
    dial_emb = emb.encode_dialogue(dial)
    return DialogueData(dial, None, dial_emb, graph)

def predict_next_cluster(dial, cur_cluster):
    return np.argmax(dial.transitions[cur_cluster])

def calc_cluster_pred_acc(dial):
    l = len(dial.second_stage_clusters)
    hits = 0
    for i in range(l - 1):
        cur_cluster = dial.second_stage_clusters[i]
        next_cluster = dial.second_stage_clusters[i+1]
        predicted_cluster = predict_next_cluster(dial, cur_cluster)
        if predicted_cluster == next_cluster:
            hits += 1
    return hits / l
