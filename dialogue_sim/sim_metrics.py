import numpy as np
from collections import defaultdict
from scipy.special import kl_div
from dialogue_sim import models
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def mean_first_stage_emb(dial1, dial2):
    emb1 = dial1.first_stage_emb.mean(axis = 0)
    emb2 = dial2.first_stage_emb.mean(axis = 0)
    sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return sim


def node_intersection(dial1, dial2):
    nodes1 = dial1.second_stage_clusters
    nodes2 = dial2.second_stage_clusters
    counts1 = defaultdict(lambda : 0)
    counts2 = defaultdict(lambda : 0)
    intersection = 0
    for node in nodes1:
        counts1[node] += 1
    for node in nodes2:
        counts2[node] += 1
    for node in counts1.keys():
        intersection += min(counts1[node], counts2[node])
    return intersection

def dice(dial1, dial2):
    nodes1 = dial1.second_stage_clusters
    nodes2 = dial2.second_stage_clusters
    counts1 = defaultdict(lambda : 0)
    counts2 = defaultdict(lambda : 0)
    s1 = s2 = 0
    intersection = 0
    for node in nodes1:
        counts1[node] += 1
        s1 += 1
    for node in nodes2:
        counts2[node] += 1
        s2 += 1
    for node in counts1.keys():
        intersection += min(counts1[node], counts2[node])
    return 2 * intersection / (s1 + s2)

def _get_dial_distr(dial):
    nodes = dial.second_stage_clusters
    tr = dial.transitions
    distr = tr[nodes].mean(axis = 0)
    return distr

def distr_intersect(dial1, dial2):
    distr1 = _get_dial_distr(dial1)
    distr2 = _get_dial_distr(dial2)
    return np.minimum(distr1, distr2).sum()
    
def JS_distance(dial1, dial2):
    distr1 = _get_dial_distr(dial1)
    distr2 = _get_dial_distr(dial2)
    M = (distr1 + distr2) / 2
    kl1 = kl_div(distr1, M).sum()
    kl2 = kl_div(distr2, M).sum()
    return np.sqrt((kl1 + kl2) / 2)

def distr_cosine(dial1, dial2):
    distr1 = _get_dial_distr(dial1)
    distr2 = _get_dial_distr(dial2)
    return distr1 @ distr2.T / (np.linalg.norm(distr1) * np.linalg.norm(distr2))

def mean_lm_embeddings(dial1, dial2):
    dial1_mean_emb = dial1.lm_embeddings.mean(axis = 0)
    dial2_mean_emb = dial2.lm_embeddings.mean(axis = 0)
    return dial1_mean_emb @ dial2_mean_emb.T / (np.linalg.norm(dial1_mean_emb) * np.linalg.norm(dial2_mean_emb))

class MetricsEval:
    def __init__(self, metrics_list = None, device = None):
        if device is None:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = device

        self.metrics = {
            'mean_first_stage_emb' : mean_first_stage_emb,
            'node_intersection' : node_intersection,
            'dice' : dice,
            'distr_intersect' : distr_intersect,
            'JS_distance' : JS_distance,
            'distr_cosine' : distr_cosine,
            'mean_lm_embeddings' : mean_lm_embeddings,
            'FFN' : lambda x, y : self.model_metric(x, y, 'FFN'),
            'GRU' : lambda x, y : self.model_metric(x, y, 'GRU'),
            'LSTM' : lambda x, y : self.model_metric(x, y, 'LSTM')
        }
        self.models = {
            'FFN' : [models.DialSimFFN, 'dialogue_sim/models/FeedForward.pt'],
            'GRU' : [models.DialSimGRU, 'dialogue_sim/models/SbertGRU.pt'],
            'LSTM' : [models.DialSimLSTM, 'dialogue_sim/models/SbertLSTM.pt'],
        }

        self.models_cached = {}

        if metrics_list is None:
            metrics_list = list(self.metrics.keys())
        self.metrics_list = metrics_list

    def get_metrics(self, dial1, dial2):
        scores = {}
        for metric in self.metrics_list:
            scores[metric] = self.metrics[metric](dial1, dial2)
        return scores
    
    def prepare_dial(self, dial):
        res = {}
        res['first_stage_emb'] = torch.Tensor(dial.first_stage_emb).to(self.device).unsqueeze(0)
        res['length'] = torch.Tensor([len(dial.first_stage_emb)]).to(self.device).unsqueeze(0)
        return res

    def model_metric(self, dial1, dial2, model_name):
        if not model_name in self.models_cached:
            mclass, loc = self.models[model_name]
            model = mclass()
            model.load_state_dict(torch.load(loc, map_location= torch.device(device)))
            model = model.to(self.device).eval()
            self.models_cached[model_name] = model
        else:
            model = self.models_cached[model_name]
        v1 = model(self.prepare_dial(dial1))
        v2 = model(self.prepare_dial(dial2))
        return (v1 @ v2.T).cpu().item()