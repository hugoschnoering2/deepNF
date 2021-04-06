
import numpy as np

import scipy.io as sio
from sklearn.preprocessing import minmax_scale

from os import listdir
from os.path import isfile, join

import torch
from torch.utils.data import TensorDataset, DataLoader

def load_data(feature_folder, annotations_folder):
  feature_files = [join(feature_folder, f) for f in listdir(feature_folder) if isfile(join(feature_folder, f))]
  annotation_files = [join(annotations_folder, f) for f in listdir(annotations_folder) if isfile(join(annotations_folder, f))]
  assert len(annotation_files) == 1, "Multiple annotation exist in the same folder"
  annotation_file = annotation_files[0]
  annotations = sio.loadmat(annotation_file)
  features = []
  if np.array([f.endswith(".mat") for f in feature_files]).all():
    for f in feature_files:
      N = sio.loadmat(f, squeeze_me=True)
      Net = N["Net"].todense()
      features.append(Net)
  elif np.array([f.endswith(".emb") for f in feature_files]).all():
    for f in feature_files:
      features.append(np.loadtxt(f))
  else:
    raise ValueError("The extension of feature files is not readable")
  return features, annotations

def processing(features):
  features_ = [minmax_scale(N) for N in features]
  return features_

def split_data(num_nodes, seed=None):
  splits = {}
  if seed is not None:
    np.random.seed(seed)
  rand = np.random.rand(num_nodes)
  splits["val"] = rand > 0.7
  train_index = rand > 0.1
  train_index[splits["val"]] = False
  splits["train"] = train_index
  splits["test"] = rand <= 0.1
  return splits

def create_dataloader(features, annotations, level, splits, batch_size):

  train_index = splits["train"]
  val_index = splits["val"]
  test_index = splits["test"]

  train_dataset = TensorDataset(*[torch.from_numpy(x[train_index].astype(np.float32)) for x in features]\
                              +[torch.from_numpy(annotations[level][train_index].astype(np.float32))])

  val_dataset = TensorDataset(*[torch.from_numpy(x[val_index].astype(np.float32)) for x in features]\
                              +[torch.from_numpy(annotations[level][val_index].astype(np.float32))])
  test_dataset = TensorDataset(*[torch.from_numpy(x[test_index].astype(np.float32)) for x in features]\
                              +[torch.from_numpy(annotations[level][test_index].astype(np.float32))])

  train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
  test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

  return train_dataloader, val_dataloader, test_dataloader
