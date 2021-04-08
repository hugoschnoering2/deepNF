
import numpy as np

import torch
import torch.nn as nn

class block(nn.Module):
  def __init__(self, dim_input, dim_output, activation, dropout=0.1):
    super().__init__()
    self.proj = nn.Linear(in_features=dim_input, out_features=dim_output)
    self.activation = activation()
    self.dropout = nn.Dropout(dropout)
  def forward(self, x):
    h = self.proj(x)
    h = self.activation(h)
    h = self.dropout(h)
    return h

class final_layer(nn.Module):
  def __init__(self, dim_input, dim_hidden, dim_output, activation, dropout=0.1):
    super().__init__()
    self.proj1 = nn.Linear(in_features=dim_input, out_features=dim_hidden)
    self.activation = activation()
    self.dropout = nn.Dropout(dropout)
    self.proj2 = nn.Linear(in_features=dim_hidden, out_features=dim_output)
  def forward(self, x):
    h = self.proj1(x)
    h = self.activation(h)
    h = self.dropout(h)
    h = self.proj2(h)
    return h

class MLP(nn.Module):
    def __init__(self, num_layers, dim_input, num_classes, dropout=0.1):
        super().__init__()
        self.layers = []
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        for i in range(num_layers-1):
            new_module = nn.Linear(in_features=dim_input, out_features=dim_input)
            self.layers.append(new_module)
            self.add_module("layer_"+str(i), new_module)
        self.final = nn.Linear(in_features=dim_input, out_features=num_classes)
    def forward(self, x):
        for layer in self.layers:
            x = self.dropout(self.relu(layer(x)))
        return self.final(x)


class MDA(nn.Module):
  def __init__(self, N, dim_input, hidden_dims, activation, dropout=0.1,
               input_noise=0., hidden_noise=0.,  classifier=None, feature_type="RWR"):
    super().__init__()

    self.input_noise = input_noise
    self.hidden_noise = hidden_noise
    self.classifier = classifier
    if classifier is not None:
      self.add_module("classifier", self.classifier)

    self.feature_type = feature_type

    self.init_embeddings = []
    for i in range(N):
      new_module = block(dim_input=dim_input, dim_output=hidden_dims[0], dropout=dropout, activation=activation)
      self.add_module("init_"+str(i), new_module)
      self.init_embeddings.append(new_module)
    middle_index = np.argmin(hidden_dims)
    self.middle_layers_c = []

    for i in range(middle_index-1):
      if i == 0:
        new_module = block(dim_input=N*hidden_dims[i], dim_output=hidden_dims[i+1], dropout=dropout, activation=activation)
      else:
        new_module = block(dim_input=hidden_dims[i], dim_output=hidden_dims[i+1], dropout=dropout, activation=activation)
      self.add_module("middle_c_"+str(i), new_module)
      self.middle_layers_c.append(new_module)

    if middle_index-1 > 0:
      i +=1
      new_module = nn.Linear(in_features=hidden_dims[i], out_features=hidden_dims[i+1])
      self.add_module("middle_c_"+str(i), new_module)
      self.middle_layers_c.append(new_module)
    else:
      new_module = nn.Linear(in_features=N*hidden_dims[0], out_features=hidden_dims[1])
      self.add_module("middle_c_"+str(i), new_module)
      self.middle_layers_c.append(new_module)

    self.activation = nn.Sequential(activation(), nn.Dropout(dropout))
    self.middle_layers_e = [self.activation]
    for i in range(middle_index, len(hidden_dims)-2):
      new_module = block(dim_input=hidden_dims[i], dim_output=hidden_dims[i+1], dropout=dropout, activation=activation)
      self.add_module("middle_e_"+str(i), new_module)
      self.middle_layers_e.append(new_module)

    self.finals = []
    for i in range(N):
      new_module = final_layer(dim_input=hidden_dims[-2], dim_hidden=hidden_dims[-1],
                               dim_output=dim_input, dropout=dropout, activation=activation)
      self.add_module("final_"+str(i), new_module)
      self.finals.append(new_module)

  def encode(self, x):
      h = torch.cat([self.init_embeddings[i](x[i]) for i in range(6)], axis=1)
      for layer in self.middle_layers_c:
        h = layer(h)
      return h

  def predict(self, x):
      assert self.classifier is not None, "No Classifier !"
      h = self.encode(x)
      logits = self.classifier(h)
      return logits

  def forward(self, x):
      if self.input_noise != 0 and self.training:
          x = [h + self.input_noise * torch.randn_like(h) for h in x]
          if self.feature_type == "RWR":
              x = [torch.clip(h, 0, 1) for h in x]
      h = self.encode(x)
      if self.hidden_noise != 0. and self.training:
        h = h + self.hidden_noise * torch.randn_like(h)
      if self.classifier is not None:
        logits = self.classifier(h)
      for layer in self.middle_layers_e:
        h = layer(h)
      outputs = [final(h) for final in self.finals]
      if self.classifier is not None:
        return outputs, logits
      return outputs
