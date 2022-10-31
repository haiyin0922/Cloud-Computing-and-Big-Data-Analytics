#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch

import copy
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split

from torch import nn, optim

import torch.nn.functional as F

import torch.utils.data as Data
import torchvision.transforms as transforms


# In[2]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[3]:


import random

my_seed = 42
random.seed(my_seed)
torch.manual_seed(my_seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(my_seed)


# In[33]:


# normal data
A_normal = pd.read_csv('./sensor_A_normal.csv')
B_normal = pd.read_csv('./sensor_B_normal.csv')
C_normal = pd.read_csv('./sensor_C_normal.csv')
D_normal = pd.read_csv('./sensor_D_normal.csv')
E_normal = pd.read_csv('./sensor_E_normal.csv')
df_normal = [A_normal, B_normal, C_normal, D_normal, E_normal]

# validation data
A_val = pd.read_csv('./sensor_A_public.csv', usecols=[0], nrows = 500)
B_val = pd.read_csv('./sensor_B_public.csv', usecols=[0], nrows = 500)
C_val = pd.read_csv('./sensor_C_public.csv', usecols=[0], nrows = 500)
D_val = pd.read_csv('./sensor_D_public.csv', usecols=[0], nrows = 500)
E_val = pd.read_csv('./sensor_E_public.csv', usecols=[0], nrows = 500)
df_val = [A_val, B_val, C_val, D_val, E_val]

# public data
A_public = pd.read_csv('./sensor_A_public.csv')
B_public = pd.read_csv('./sensor_B_public.csv')
C_public = pd.read_csv('./sensor_C_public.csv')
D_public = pd.read_csv('./sensor_D_public.csv')
E_public = pd.read_csv('./sensor_E_public.csv')
df_public = [A_public, B_public, C_public, D_public, E_public]

# private data
A_private = pd.read_csv('./sensor_A_private.csv')
B_private = pd.read_csv('./sensor_B_private.csv')
C_private = pd.read_csv('./sensor_C_private.csv')
D_private = pd.read_csv('./sensor_D_private.csv')
E_private = pd.read_csv('./sensor_E_private.csv')
df_private = [A_private, B_private, C_private, D_private, E_private]


# In[5]:


class DemoDatasetLSTM(Data.Dataset):
    """
        Support class for the loading and batching of sequences of samples

        Args:
            dataset (Tensor): Tensor containing all the samples
            sequence_length (int): length of the analyzed sequence by the LSTM
            transforms (object torchvision.transform): Pytorch's transforms used to process the data
    """
    ##  Constructor
    def __init__(self, dataset, sequence_length=1, transforms=None):
        self.dataset = dataset
        self.seq_len = sequence_length
        self.transforms = transforms

    ##  Override total dataset's length getter
    def __len__(self):
        return self.dataset.__len__()

    ##  Override single items' getter
    def __getitem__(self, idx):
        if idx + self.seq_len > self.__len__():
            if self.transforms is not None:
                item = torch.zeros(self.seq_len, self.dataset[0].__len__())
                item[:self.__len__()-idx] = self.transforms(self.dataset[idx:])
                return item, item
            else:
                item = []
                item[:self.__len__()-idx] = self.dataset[idx:]
                return item, item
        else:
            if self.transforms is not None:
                return self.transforms(self.dataset[idx:idx+self.seq_len]), self.transforms(self.dataset[idx:idx+self.seq_len])
            else:
                return self.dataset[idx:idx+self.seq_len], self.dataset[idx:idx+self.seq_len]


# In[6]:


###   Helper for transforming the data from a list to Tensor

def listToTensor(list):
    tensor = torch.empty(list.__len__(), list[0].__len__())
    for i in range(list.__len__()):
        tensor[i, :] = torch.from_numpy(list[i])
        
    return tensor


# In[7]:


# Parameters

seq_len = 100
n_features = 1
batch_size = 64


# In[8]:


def create_dataset(df):

    data_transform = transforms.Lambda(lambda x: listToTensor(x))
    
    dataset = []
    for d in df:
        df_data = np.array(d.iloc[:, 0].values).astype(float).reshape(-1, 1)
        dataset.append(DemoDatasetLSTM(df_data, seq_len, transforms=data_transform))
    
    data_loader = []
    for d in dataset:
        data_loader.append(Data.DataLoader(d, batch_size, shuffle=False))

    for data1 in data_loader:
        for data2 in data1:
            x, _ = data2
            print(x)
            print('\n')
    
    return data_loader


# In[9]:


class LSTMEncoder(nn.Module):
    
    def __init__(self, seq_len, n_features, embedding_dim):
        super(LSTMEncoder, self).__init__()
        
        # Parameters
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.hidden_dim = 2*embedding_dim
        
        # Neural Network Layers
        self.lstm1 = nn.LSTM(self.n_features, self.hidden_dim, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(self.hidden_dim, self.embedding_dim, num_layers=1, batch_first=True)
    
    def forward(self, i): 
        i, _ = self.lstm1(i)               # from (batch, seq_len, n_features) to (batch, seq_len, hidden_dim)
        i, (hidden_n, _) = self.lstm2(i)   # from (batch, seq_len, hidden_dim) to (batch, seq_len, embedding_dim)
        
        return hidden_n                    # hidden_n shape: (num_layers*num_directions, batch, embedding_dim)


# In[10]:


class LSTMDecoder(nn.Module):

    def __init__(self, seq_len, embedding_dim, n_features=1):
        super(LSTMDecoder, self).__init__()

        # Parameters
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.hidden_dim = 2*embedding_dim
        self.n_features = n_features
        
        # Neural Network Layers
        self.lstm1 = nn.LSTM(self.embedding_dim, self.embedding_dim, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=1, batch_first=True)
        self.output_layer = nn.Linear(self.hidden_dim, n_features)
        
    def forward(self, i):
        # Do padding
        i = i.repeat(self.seq_len, 1, 1)                       # repeat (1, embedding_dim) to (seq_len, embedding_dim)
        i = i.reshape((-1, self.seq_len, self.embedding_dim))  # reshape to (batch, seq_len, embedding_dim)
        
        # Traverse neural layers
        i, _ = self.lstm1(i)      # from (batch, seq_len, embedding_dim) to (batch, seq_len, embedding_dim)
        i, _ = self.lstm2(i)      # from (batch, seq_len, embedding_dim) to (batch, seq_len, hidden_dim)
        i = self.output_layer(i)  # from (batch, seq_len, hidden_dim) to (batch, seq_len, n_features)
        
        return i


# In[11]:


class LSTMAutoencoder(nn.Module):
    
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = LSTMEncoder(seq_len, n_features, embedding_dim).to(device)
        self.decoder = LSTMDecoder(seq_len, embedding_dim, n_features).to(device)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def forward(self, i):
        i = self.encoder(i)
        i = self.decoder(i)
        
        return i


# In[12]:


def train_model(model, train_dataset, val_dataset, n_epochs):

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss(reduction='sum').to(device)
    history = dict(train=[], val=[])
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0

    for epoch in range(1, n_epochs + 1):
        model = model.train()
        train_losses = []

        for seq_true, _ in train_dataset:
            optimizer.zero_grad()
            seq_true = seq_true.to(device)
            seq_pred = model(seq_true)
            loss = criterion(seq_pred, seq_true)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        val_losses = []
        model = model.eval()

        with torch.no_grad():
            for _, seq_true in val_dataset:
                seq_true = seq_true.to(device)
                seq_pred = model(seq_true)
                loss = criterion(seq_pred, seq_true)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history['train'].append(train_loss)
        history['val'].append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')

    model.load_state_dict(best_model_wts)

    return model.eval(), history


# In[30]:


def predict(model, dataset):

    predictions, losses = np.array([]), np.array([])
    criterion = nn.L1Loss(reduction='sum').to(device)
    
    with torch.no_grad():
        model = model.eval()
        for seq_true, _ in dataset:
            seq_true = seq_true.to(device)
            seq_pred = model(seq_true)
            loss = criterion(seq_pred, seq_true)
            predictions = np.append(predictions, seq_pred[:, -1, :].cpu().numpy())
            losses = np.append(losses, loss.item())
    
    return predictions, losses


# In[14]:


normal_dataset = create_dataset(df_normal)
val_dataset = create_dataset(df_val)


# In[17]:


for i in range(5):
    model = LSTMAutoencoder(seq_len, n_features, 64)
    model = model.to(device)
    
    model, history = train_model(
      model, 
      normal_dataset[i], 
      val_dataset[i], 
      n_epochs=150
    )
    
    torch.save(model, f'./model/sensor_model_{i}.pth')


# In[42]:


target = []
for df in df_public:
    target.append(df.iloc[:, -1].values)
    df = df.drop(['label'], axis=1)
    print(df)


# In[43]:


public_dataset = create_dataset(df_public)
private_dataset = create_dataset(df_private)


# In[47]:


from sklearn.metrics import roc_auc_score


# In[53]:


submission = np.array([])
for i in range(5):
    model = torch.load(f'./model/sensor_model_{i}.pth')
    reconstructed_val, losses_val = predict(model, public_dataset[i]) 
    print(np.array(reconstructed_val).shape)
    origin_data = df_public[i].iloc[:, 0].values
    print(origin_data)
    reconstructed_error = np.abs((reconstructed_val - origin_data))
    print("reconstructed_error:", reconstructed_error.shape)
    print("AUC score:", roc_auc_score(target[i], reconstructed_error))
    submission = np.append(submission, reconstructed_error)
    #submission = np.append(submission, target[i])


# In[54]:


for i in range(5):
    model = torch.load(f'./model/sensor_model_{i}.pth')
    reconstructed_val, losses_val = predict(model, private_dataset[i])
    print(reconstructed_val.shape)
    origin_data = df_private[i].iloc[:, 0].values
    print(origin_data)
    reconstructed_error = np.abs((reconstructed_val - origin_data))
    print("reconstructed_error:", reconstructed_error.shape)
    #print("AUC score:", roc_auc_score(target[i], reconstructed_error))
    submission = np.append(submission, reconstructed_error)
    
print(submission.shape)


# In[55]:


df_submission = pd.DataFrame(submission, columns=['pred'])
df_submission.insert(0, 'id', df_submission.index)
df_submission
df_submission.to_csv('submission.csv', index=False)


# In[ ]:




