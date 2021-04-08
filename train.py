
import argparse
import yaml

import numpy as np
import torch
import torch.nn as nn

from data_utils import load_data, processing, split_data, create_dataloader, minmax_scale
from metrics import evaluate_performance
from models import MDA
from utils import YamlNamespace

def train_ae_one_epoch(model, optimizer, dataloader, ae_criterion,
                       classifier_criterion, kappa=1., multitasking=False):
  model.train()
  for batch in dataloader:
    model.zero_grad()
    input = [b.to(device) for b in batch[:-1]]
    labels = batch[-1].to(device)
    if model.classifier is None:
        ae_output = model(input)
    elif multitasking:
      ae_output, classifier_logits = model(input)
      classifier_loss = classifier_criterion(classifier_logits, labels)
    else:
      ae_output, _ = model(input)
    ae_loss = torch.tensor([0.]).to(device)
    for i in range(len(input)):
      ae_loss += ae_criterion(ae_output[i], input[i])
    if (model.classifier is not None) and multitasking:
      loss = ae_loss + kappa * classifier_loss
    else:
      loss = ae_loss
    loss.backward()
    optimizer.step()

def evaluate_ae(model, dataloader, ae_criterion, classifier_criterion, multitasking=False):
  model.eval()
  with torch.no_grad():
    ae_loss = 0.
    classifier_loss = 0.
    for batch in dataloader:
      input = [b.to(device) for b in batch[:-1]]
      labels = batch[-1].to(device)
      if model.classifier is None:
          ae_output = model(input)
      elif multitasking:
        ae_output, classifier_logits = model(input)
        classifier_loss += classifier_criterion(classifier_logits, labels).item()
      else:
        ae_output, _ = model(input)
      for i in range(len(input)):
        ae_loss += ae_criterion(ae_output[i], batch[i].to(device)).item()
    return ae_loss, classifier_loss

def train_classifier_one_epoch(model, optimizer, dataloader, criterion):
  model.train()
  for batch in dataloader:
    model.zero_grad()
    input = [b.to(device) for b in batch[:-1]]
    labels = batch[-1].to(device)
    logits = model.predict(input)
    classifier_loss = criterion(logits, labels)
    classifier_loss.backward()
    optimizer.step()

def evaluate_classifier(model, dataloader, criterion):
  model.eval()
  with torch.no_grad():
    classifier_loss = 0.
    for batch in dataloader:
      input = [b.to(device) for b in batch[:-1]]
      labels = batch[-1].to(device)
      logits = model.predict(input)
      classifier_loss += criterion(logits, labels).item()
  return classifier_loss

def evaluate_predictions(model, dataloader):
  model.eval()
  y_test_list = []
  y_score_list = []
  y_pred_list = []
  with torch.no_grad():
    for batch in dataloader:
      input = [b.to(device) for b in batch[:-1]]
      label = batch[-1].to(device)
      logits = model.predict(input)
      y_test_list.append(label.detach().cpu().numpy())
      y_score_list.append(torch.sigmoid(logits).detach().cpu().numpy())
      y_pred_list.append(np.where(logits.detach().cpu().numpy() >= 0, 1, 0))
  perf = evaluate_performance(np.concatenate(y_test_list, axis=0),
                              np.concatenate(y_score_list, axis=0),
                              np.concatenate(y_pred_list, axis=0))
  return perf

def _parse_args():
    parser = argparse.ArgumentParser("Training script")
    parser.add_argument("--config", "-c", type=str, required=True, help="The YAML config file")
    cli_args = parser.parse_args()
    with open(cli_args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = YamlNamespace(config)
    return config

if __name__ == "__main__":

    config = _parse_args()
    if config.seed is not None:
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
    config.activation = {"relu" : nn.ReLU, "sigmoid" : nn.Sigmoid}[config.activation]

    print("### Loading the data")
    X, A = load_data(config.feature_folder, config.annotation_folder)
    X = processing(X, config.features)
    splits = split_data(X[0].shape[0])
    train_dataloader, val_dataloader, test_dataloader = \
    create_dataloader(X, A, "level"+str(config.level), splits, config.batch_size)

    print("### Building the model")
    if config.features == "RWR":
        dim_input = 6400
        hidden_dims = [2000, 600, 2000]
    elif config.features == "LINE" or config.features == "SDNE":
        dim_input = 128
        hidden_dims = [96, 64, 96]
    else:
        raise ValueError("These features are not available")
    if config.classifier == "nn":
        classifier = nn.Linear(np.min(hidden_dims),
                               A["level"+str(config.level)].shape[1])
    elif config.classifier == "svm":
        classifier = None
    else:
        raise ValueError("This classifier is not available")
    model = MDA(N=len(X),
                dim_input=dim_input,
                hidden_dims=hidden_dims,
                hidden_noise=config.hidden_noise,
                input_noise=config.input_noise,
                classifier=classifier,
                feature_type=config.features,
                activation=config.activation)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    classifier_criterion = nn.BCEWithLogitsLoss()
    if config.features == "RWR":
        ae_criterion = classifier_criterion
    else:
        ae_criterion = nn.MSELoss()

    print("### Starting the training -- Auto-Encoder")
    assert config.classifier == "nn" or not config.multitasking

    for _ in range(config.epochs):
        train_ae_one_epoch(model, optimizer, train_dataloader, ae_criterion,
                           classifier_criterion, config.kappa, config.multitasking)
        ae_loss, classifier_loss = evaluate_ae(model, val_dataloader, ae_criterion,
                                               classifier_criterion, config.multitasking)
        if config.multitasking:
            F1_train = evaluate_predictions(model, train_dataloader)["F1"]
            F1_val = evaluate_predictions(model, val_dataloader)["F1"]
            print("F1 train dataset : {0:.3f}, F1 val dataset : {1:.3f}".format(F1_train, F1_val))

    if config.classifier == "nn" and not(config.multitasking):
        print("### Starting the training -- Linear Classifier")
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=config.lr)
        for _ in range(config.epochs):
            train_classifier_one_epoch(model, optimizer, train_dataloader, classifier_criterion)
            classifier_loss = evaluate_classifier(model, val_dataloader, classifier_criterion)
            F1_train = evaluate_predictions(model, train_dataloader)["F1"]
            F1_val = evaluate_predictions(model, val_dataloader)["F1"]
            print("F1 train dataset : {0:.3f}, F1 val dataset : {1:.3f}".format(F1_train, F1_val))

    print("### Evaluating the performance")
    if config.classifier == "svm":
        from svm import cross_validation
        model.eval()
        input = [torch.from_numpy(x.astype(np.float32)).to(device) for x in X]
        with torch.no_grad():
            low_dim_embeddings = torch.sigmoid(model.encode(input)).detach().cpu().numpy()
        perf = cross_validation(minmax_scale(low_dim_embeddings), A["level"+str(config.level)])
    else:
        perf = evaluate_predictions(model, test_dataloader)
    print("F1 test dataset : {0:.3f}".format(perf["F1"]))
