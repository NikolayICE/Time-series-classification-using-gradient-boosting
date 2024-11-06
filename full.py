import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import intel_extension_for_pytorch as ipex
import scipy
import random
import datetime as dt
from sklearn.preprocessing import LabelEncoder
import sys, os
from xgboost import XGBClassifier
warnings.filterwarnings('ignore')

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
lb_enc_label = LabelEncoder()
df_train.iloc[:, 1] = lb_enc_label.fit_transform(df_train.iloc[:, 1])
lb_enc_sex = LabelEncoder()
df_train.iloc[:, 2] = lb_enc_sex.fit_transform(df_train.iloc[:, 2])
lb_enc_patient = LabelEncoder()
df_train.iloc[:, 4] = lb_enc_patient.fit_transform(df_train.iloc[:, 4])
df_test.iloc[:, 1] = lb_enc_sex.transform(df_test.iloc[:, 1])
df_test.iloc[:, 3] = lb_enc_patient.transform(df_test.iloc[:, 3])

train_signals_t, train_labels = [], []
for obj in tqdm(range(len(df_train))):
    filename = df_train.iloc[obj, 0]
    data = pd.read_csv(os.path.join('data', filename))
    train_signals_t.append(data.to_numpy()[:100])
    label = [0, 0, 0, 0, 0, 0]
    real_label = df_train.iloc[obj, 1]
    label[real_label] = 1
    train_labels.append(label)
train_labels = np.array(train_labels)

train_signals = []
for obj in range(len(train_signals_t)):
    t_signal = np.transpose(train_signals_t[obj])
    train_signals.append(t_signal)
train_signals = np.array(train_signals)

signals_train = []
window_size = 3
for i in tqdm(range(len(train_signals))):
    tmp_signal = []
    for j in range(len(train_signals[i])):
        signal = [train_signals[i, j, 0]]
        for k in range(len(train_signals[i, j]) - window_size + 1):
            signal.append(train_signals[i, j, k:k + window_size].mean())
        signal.append(train_signals[i, j, -1])
        signal = np.array(signal)
        tmp_signal.append(signal)
    tmp_signal = np.array(tmp_signal)
    signals_train.append(tmp_signal)

signals_train = np.array(signals_train)
print(signals_train.shape)

x_train, x_test, y_train, y_test = signals_train[180:], signals_train[:180], train_labels[180:], train_labels[:180]

class CoolDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.Tensor(x)
        self.y = torch.from_numpy(y).float()
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self, index):
        return (self.x[index], self.y[index])

dataset_train = CoolDataset(x_train, y_train)
dataset_test = CoolDataset(x_test, y_test)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.drop1 = nn.Dropout(0.3)
        self.conv1 = nn.Conv1d(64, 128, 4, 2, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU(True)
        self.drop2 = nn.Dropout(0.3)
        self.conv2 = nn.Conv1d(128, 256, 4, 2, 1)
        self.bn2 = nn.BatchNorm1d(256)
        self.elu1 = nn.ELU()
        self.drop3 = nn.Dropout(0.3)
        self.conv3 = nn.Conv1d(256, 512, 4, 2, 1)
        self.bn3 = nn.BatchNorm1d(512)
        self.elu2 = nn.ELU()
        self.conv4 = nn.Conv1d(512, 1024, 4, 2, 1)
        self.bn4 = nn.BatchNorm1d(1024)
        self.elu3 = nn.ELU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(6144, 100)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(100, 6)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.drop1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.drop2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.elu1(x)
        x = self.drop3(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.elu2(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.elu3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.sig(x)
        return x


class Net_light(nn.Module):
    def __init__(self):
        super(Net_light, self).__init__()
        self.drop1 = nn.Dropout(0.3)
        self.conv1 = nn.Conv1d(64, 128, 4, 2, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU(True)
        self.drop2 = nn.Dropout(0.3)
        self.conv2 = nn.Conv1d(128, 256, 4, 2, 1)
        self.bn2 = nn.BatchNorm1d(256)
        self.elu1 = nn.ELU()
        self.drop3 = nn.Dropout(0.3)
        self.conv3 = nn.Conv1d(256, 512, 4, 2, 1)
        self.bn3 = nn.BatchNorm1d(512)
        self.elu2 = nn.ELU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(6144, 100)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(100, 6)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.drop1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.drop2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.elu1(x)
        x = self.drop3(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.elu2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.sig(x)
        return x

model = Net()
model_light = Net_light()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)
optimizer_light = torch.optim.Adam(model_light.parameters(), lr=0.0001, weight_decay=0.001)
scheduler_cos = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 0.7)
scheduler_exp = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.6)
criterion = torch.nn.BCELoss()
model, optimizer = ipex.optimize(model=model, optimizer=optimizer)
model_light, optimizer_light = ipex.optimize(model=model_light, optimizer=optimizer_light)

num_epoch = 1000
batch_size = 32

dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

for epoch in range(num_epoch):
    mean_loss, mean_acc, b_tr, b_ts = 0, 0, 0, 0
    for batch_idx, (data, target) in enumerate(dataloader_train):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        mean_loss += loss
        loss.backward()
        optimizer.step()
        output = output.detach().numpy()
        target = target.numpy()
        labels_true, labels_pred = [i.argmax() for i in target], [j.argmax() for j in output]
        acc = accuracy_score(labels_pred, labels_true)
        mean_acc += acc
        b_tr += 1
    acc_test, loss_test = 0, 0
    for batch_idx, (data, target) in enumerate(dataloader_test):
        with torch.no_grad():
            output = model(data)
            loss = criterion(output, target)
            output = output.detach().numpy()
            target = target.numpy()
            labels_true, labels_pred = [i.argmax() for i in target], [j.argmax() for j in output]
            acc = accuracy_score(labels_pred, labels_true)
            acc_test += acc
            loss_test += loss
            b_ts += 1

    if epoch % 10 == 0:
        print(f'{epoch}/{num_epoch}  loss_train: {mean_loss / b_tr}  acc_train: {mean_acc / b_tr}')
        print(f'loss_test: {loss_test / b_ts}  acc_test: {acc_test / b_ts}')

num_epoch = 1000
batch_size = 32

x_t, x_te, y_t, y_te = signals_train[:650], signals_train[650:], train_labels[:650], train_labels[650:]
dataset_t = CoolDataset(x_t, y_t)
dataset_te = CoolDataset(x_te, y_te)

dataloader_t = DataLoader(dataset_t, batch_size=batch_size, shuffle=True)
dataloader_te = DataLoader(dataset_te, batch_size=batch_size, shuffle=True)

for epoch in range(num_epoch):
    mean_loss, mean_acc, b_tr, b_ts = 0, 0, 0, 0
    for batch_idx, (data, target) in enumerate(dataloader_t):
        optimizer_light.zero_grad()
        output = model_light(data)
        loss = criterion(output, target)
        mean_loss += loss
        loss.backward()
        optimizer_light.step()
        output = output.detach().numpy()
        target = target.numpy()
        labels_true, labels_pred = [i.argmax() for i in target], [j.argmax() for j in output]
        acc = accuracy_score(labels_pred, labels_true)
        mean_acc += acc
        b_tr += 1
    acc_test, loss_test = 0, 0
    for batch_idx, (data, target) in enumerate(dataloader_te):
        with torch.no_grad():
            output = model_light(data)
            loss = criterion(output, target)
            output = output.detach().numpy()
            target = target.numpy()
            labels_true, labels_pred = [i.argmax() for i in target], [j.argmax() for j in output]
            acc = accuracy_score(labels_pred, labels_true)
            acc_test += acc
            loss_test += loss
            b_ts += 1

    if epoch % 10 == 0:
        print(f'{epoch}/{num_epoch}  loss_train: {mean_loss / b_tr}  acc_train: {mean_acc / b_tr}')
        print(f'loss_test: {loss_test / b_ts}  acc_test: {acc_test / b_ts}')

test_signals_t = []
for obj in tqdm(range(len(df_test))):
    filename = df_test.iloc[obj, 0]
    data = pd.read_csv(os.path.join('data', filename))
    test_signals_t.append(data.to_numpy()[:100])
test_signals = []
for obj in range(len(test_signals_t)):
    t_signal = np.transpose(test_signals_t[obj])
    test_signals.append(t_signal)
test_signals = np.array(test_signals)

signals_test = []
window_size = 3
for i in tqdm(range(len(test_signals))):
    tmp_signal = []
    for j in range(len(test_signals[i])):
        signal = [test_signals[i, j, 0]]
        for k in range(len(test_signals[i, j]) - window_size + 1):
            signal.append(test_signals[i, j, k:k + window_size].mean())
        signal.append(test_signals[i, j, -1])
        signal = np.array(signal)
        tmp_signal.append(signal)
    tmp_signal = np.array(tmp_signal)
    signals_test.append(tmp_signal)

signals_test = np.array(signals_test)
print(signals_test.shape)

torch.save(model.state_dict(), 'model.pt')
torch.save(model_light.state_dict(), 'model_light.pt')
model.load_state_dict(torch.load('model.pt'))
model_light.load_state_dict(torch.load('model_light.pt'))
signals_test = torch.Tensor(signals_test)
output_test = model(signals_test)
output_test = output_test.detach().numpy()
output_test_light = model_light(signals_test)
output_test_light = output_test_light.detach().numpy()
params = {
    'max_depth': [3, 4, 5, 6],
    'booster': ['gbtree', 'gblinear', 'dart'],
    'eta': [0.01, 0.02, 0.03],
    'reg_lambda': [0.01, 0.03, 0.05],
    'tree_method': ['auto', 'exact', 'approx', 'hist']
}
xgb = XGBClassifier(
    booster = 'gbtree',
    colsample_bylevel = 0.5,
    colsample_bytree = 0.7,
    subsampple = 0.8,
    gamma = 0.0,
    learning_rate = 0.02,
    max_depth = 10,
    n_estimators = 1000,
    n_jobs = -1,
    seed = 555)

x_xgb, y_xgb = [], []
proba = model(torch.Tensor(signals_train))
proba = proba.detach().numpy()
proba_light = model_light(torch.Tensor(signals_train))
proba_light = proba_light.detach().numpy()
for i in range(len(df_train)):
    line = df_train.iloc[i, 2:].to_numpy()
    line = np.append(line, proba[i])
    line = np.append(line, proba_light[i])
    x_xgb.append(line)
    y_xgb.append(df_train.iloc[i, 1])

x_train_xgb, x_test_xgb, y_train_xgb, y_test_xgb = train_test_split(x_xgb, y_xgb, test_size=0.2)
xgb.fit(x_train_xgb, y_train_xgb, eval_set=[(x_test_xgb, y_test_xgb)])

output_xgb = xgb.predict(x_test_xgb)
score = accuracy_score(output_xgb, y_test_xgb)

data_test = []
for i in range(len(df_test)):
    line = df_test.iloc[i, 1:].to_numpy()
    line = np.append(line, output_test[i])
    line = np.append(line, output_test_light[i])
    data_test.append(line)

test_labels = xgb.predict(data_test)
test_labels = lb_enc_label.inverse_transform(test_labels)

ans = pd.DataFrame(columns=['path', 'pred'])
filenames = []
for i in range(len(df_test)):
    filename = df_test.iloc[i, 0]
    filenames.append(filename)
ans['path'] = filenames
ans['pred'] = test_labels
ans.to_csv('ans.csv', index=False)