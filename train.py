from logistic_regression import LogisticRegression
from correlated_datasets import CorrelatedDataset, CorrelatedMissingDataset

import argparse
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 
from sklearn.metrics import classification_report

from tqdm import tqdm

BATCH_SIZE  = 32
POS_WEIGHT = torch.tensor([20.0])



class WeightedBCELoss(torch.nn.Module):
    def __init__(self, weight_factor):
        super(WeightedBCELoss, self).__init__()
        self.weight_factor = weight_factor

    def forward(self, logits, targets):
        
        logits = torch.sigmoid(logits)

        loss = - (targets * torch.log(logits + 1e-8) + (1 - targets) * torch.log(1 - logits + 1e-8))

        pos_labels = targets == 1

        loss[pos_labels] *= self.weight_factor
        loss = torch.mean(loss)
        return loss

def train_baseline(num_epochs: int, num_classes: int = 20, data_dimention: int = 100, data_latent_correlation_dimention: int = 5):

  training_data = CorrelatedDataset(C=num_classes, d=data_dimention, d_prime=data_latent_correlation_dimention, num_samples=200)
  train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)

  # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  device = 'cpu'
  # criterion = nn.BCEWithLogitsLoss(weight=WEIGHT).to(device)
  # criterion = nn.BCEWithLogitsLoss(pos_weight=POS_WEIGHT).to(device)
  criterion = WeightedBCELoss(POS_WEIGHT.item())
  sigmoid = nn.Sigmoid()
  classifiers = [LogisticRegression(data_dimention, 1).to(device) for _ in range(num_classes)]
  optimizers = [optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4, dampening=0.0025) for net in classifiers]
  total_losses = []
  

  for epoch in tqdm(range(num_epochs), desc=f"Training progress", colour="#00ff00"):
    losses = []
    training_samples = []

    num_samples = 0
    num_positive = [0] * num_classes
    for batch_idx, (inputs, targets) in enumerate(tqdm(train_dataloader, leave=False, desc=f"Epoch {epoch + 1}/{num_epochs}", colour="#005500")):
      training_samples.append((inputs, targets))
      inputs = inputs.to(device)
      targets = targets.to(device)
      num_samples += targets.shape[0]
      
      for c in range(num_classes):
        num_positive[c] += torch.sum(targets[:, c]).item()

        optimizers[c].zero_grad()
        outputs = classifiers[c](inputs)

        if torch.sum(targets[:, c]) == 0:
          continue

        # target = nn.functional.one_hot(targets[:, c].to(torch.int64)).double()
        loss = criterion(outputs, targets[:, c].unsqueeze(1))
        loss.backward()
        optimizers[c].step()
        losses.append(loss.cpu().item())
    total_losses.append(sum(losses) / len(losses))

    # print(num_samples)
    # print(num_positive)
    # print()

  plt.plot(total_losses)
  plt.savefig('losses.png')


  y_pred = []
  y_true = []

  for batch_idx, (inputs, targets) in enumerate(train_dataloader):
    inputs = inputs.to(device)
    targets = targets.to(device)
    
    for c in range(num_classes):
      outputs = classifiers[c](inputs)
      predicted = sigmoid(outputs).round().cpu().detach().numpy()
      expected  = targets[:, c].cpu().detach().numpy()

      y_pred.append(predicted)
      y_true.append(expected)
      # correct = np.sum(pred_outputs == expected_outputs)

      # pos_mask = expected_outputs == 1
      # correct_positive = np.sum(pred_outputs[pos_mask] == 1)
      # print(pred_outputs)
      # print(expected_outputs)
      # print()
      # print(f'Correct: {correct} Correct Positive: {correct_positive}')
      # print()
      # print()
      # print(pred_outputs == targets.cpu().detach().numpy())

  y_pred = np.concatenate(y_pred)
  y_true = np.concatenate(y_true)
  print(classification_report(y_true, y_pred))


  

# def test()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--n_epochs', type=int, default=25)
  args = parser.parse_args()
  train_baseline(args.n_epochs)
