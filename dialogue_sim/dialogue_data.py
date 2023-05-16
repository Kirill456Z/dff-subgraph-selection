from dataclasses import dataclass
import numpy as np

@dataclass
class DialogueData:
    def _get_second_stage_emb(self, dial, data, embeddings, graph):
        start_idx = data.get_dialog_start_idx(dial)
        self.first_stage_clusters = []
        self.first_stage_emb = []
        self.second_stage_clusters = []
        self.lm_embeddings = []
        for i in range(len(dial.utterances)):
            lm_embedd = embeddings[start_idx + i, :].ravel()
            self.lm_embeddings.append(lm_embedd)
            cluster = graph.one_stage_clustering.predict_cluster(lm_embedd, dial[i], dial).id
            self.first_stage_emb.append(graph.cluster_embeddings[cluster])
            self.first_stage_clusters.append(cluster)
            self.second_stage_clusters.append(graph.cluster_kmeans_labels[0][cluster])
        self.lm_embeddings = np.array(self.lm_embeddings)
        self.first_stage_emb = np.array(self.first_stage_emb)
    
    def __getitem__(self, key):
        return self.key_to_attr[key]

    def __init__(self, dial, data, embeddings, graph):
        utts = [str(utt) for utt in dial.utterances]
        r =  ".".join(utts) + "."
        self.str = r
        self.services = dial.meta['services']
        self._get_second_stage_emb(dial, data,embeddings, graph)
        self.transitions = np.array(graph.get_transitions())
        self.key_to_attr = {
            'first_stage_emb' : self.first_stage_emb,
            'length' : len(self.first_stage_emb),
        }