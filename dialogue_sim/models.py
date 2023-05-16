import torch.nn as nn
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DialSimFFN(nn.Module):
  def __init__(self):
    super().__init__()
    self.l1 = nn.Linear(in_features = 8, out_features = 64)
    self.act = nn.GELU()
    self.l2 = nn.Linear(in_features = 64, out_features = 16)
  
  def forward(self, data):
    emb = data['first_stage_emb'].float()
    l = data['length']
    x = self.l1(emb.sum(axis = 1) / l.reshape(-1, 1))
    x = self.act(x)
    return self.l2(x)


class DialSimGRU(nn.Module):
  def __init__(self):
    super().__init__()
    self.rnn = nn.GRU(input_size = 8, hidden_size = 64, num_layers = 2, batch_first = True, bidirectional = True)
  
  def forward(self, data):
    lengths = data['length'].cpu()
    emb = data['first_stage_emb'].to(device).float()
    if len(emb) > 1:
      ps = nn.utils.rnn.pack_padded_sequence(emb, lengths, batch_first = True, enforce_sorted = False)
    else:
      ps = emb
    return self.rnn(ps)[1][0]


class DialSimLSTM(nn.Module):
  def __init__(self):
    super().__init__()
    self.rnn = nn.LSTM(input_size = 8, hidden_size = 64, num_layers = 1, bidirectional = True, batch_first = True)
  
  def forward(self, data):
    lengths = data['length'].cpu()
    emb = data['first_stage_emb'].to(device).float()
    if len(emb) > 1:
      ps = nn.utils.rnn.pack_padded_sequence(emb, lengths, batch_first = True, enforce_sorted = False)
    else:
      ps = emb
    return torch.cat(tuple(self.rnn(ps)[1][1]), dim = -1)
  
model = DialSimLSTM().to(device)