import numpy as np
import pandas as pd
import wave
import librosa
from python_speech_features import *
import pickle

import torch 
from torch.utils import data # 获取迭代数据
from torch.autograd import Variable # 获取变量
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error
from torch.nn import functional as F

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

prefix = '/Users/apple/Downloads/depression/'
with open(prefix+'avid_info.pkl', 'rb') as f:
    split_info = pickle.load(f)

train_split_num = split_info['train'][0]
train_split_label = [int(x) for x in split_info['train'][1]]
test_split_num = split_info['test'][0]
test_split_label = [int(x) for x in split_info['test'][1]]
# train_split_label = np.hstack((train_split_label, dev_split_label))
# train_split_num = np.hstack((train_split_num, dev_split_num))

def extract_features(number, audio_features, mode):    
    wavefile = wave.open(prefix+'AViD/Audio/{1}/Trim/{0}.wav'.format(number, mode))
    sr = wavefile.getframerate()
    nframes = wavefile.getnframes()
    wave_data = np.frombuffer(wavefile.readframes(nframes), dtype=np.short).astype(np.float)    
    
    if len(wave_data) < sr*15:
        wave_data = np.hstack((wave_data, [1e-2]*(sr*15-len(wave_data))))
        
    # 1分钟
    clip = sr*1*15
    melspec = librosa.feature.melspectrogram(wave_data[:clip], n_mels=80, sr=sr)
    audio_features.append(melspec)
    if sr == 32000:
        print(prefix+'AViD/Audio/{1}/Trim/{0}.wav'.format(number, mode))
    
# training set
audio_features_train = []
# test set
audio_features_test = []

# # training set
# for index in range(len(train_split_num)):
#     extract_features(train_split_num[index], audio_features_train, 'Training')

# # test set
# for index in range(len(test_split_num)):
#     extract_features(test_split_num[index], audio_features_test, 'Testing')

# print(np.shape(audio_features_train), np.shape(audio_features_test))

# print("Saving npz file locally...")
# np.savez(prefix+'data/audio/train_samples_reg_avid.npz', audio_features_train)
# np.savez(prefix+'data/audio/train_labels_reg_avid.npz', train_split_label)
# np.savez(prefix+'data/audio/test_samples_reg_avid.npz', audio_features_test)
# np.savez(prefix+'data/audio/test_labels_reg_avid.npz', test_split_label)

audio_features_train = np.load(prefix+'data/audio/train_samples_reg_avid.npz', allow_pickle=True)['arr_0']
audio_ctargets_train = np.load(prefix+'data/audio/train_labels_reg_avid.npz', allow_pickle=True)['arr_0']
audio_features_test = np.load(prefix+'data/audio/test_samples_reg_avid.npz', allow_pickle=True)['arr_0']
audio_ctargets_test = np.load(prefix+'data/audio/test_labels_reg_avid.npz', allow_pickle=True)['arr_0']

config = {
    'num_classes': 1,
    'dropout': 0.5,
    'rnn_layers': 2,
    'embedding_size': 80,
    'batch_size': 2,
    'epochs': 30,
    'learning_rate': 1e-3,
    'cuda': False,
}

X_train = np.array(audio_features_train)
Y_train = np.array(audio_ctargets_train)
X_test = np.array(audio_features_test)
Y_test = np.array(audio_ctargets_test)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train = np.array([(X - X.min()) / (X.max() - X.min()) if X.max() != X.min() else X for X in X_train ])
X_test = np.array([(X - X.min()) / (X.max() - X.min()) if X.max() != X.min() else X for X in X_test ])

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv2d_1 = nn.Conv2d(1, 32, (1,7), 1)
        self.conv2d_2 = nn.Conv2d(32, 32, (1,7), 2)
        self.dense_1 = nn.Linear(87360, 128)
        self.dense_2 = nn.Linear(128, 128)
        self.dense_3 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = F.relu(self.conv2d_1(x))
        x = F.max_pool2d(x, (4, 3), (1, 3))
        x = F.relu(self.conv2d_2(x))
        x = F.max_pool2d(x, (1, 3), (1, 3))
#         flatten in keras
        x = x.permute((0, 2, 3, 1))
        x = x.contiguous().view(-1, 87360)
        x = F.relu(self.dense_1(x))
        x = F.relu(self.dense_2(x))
        x = self.dropout(x)
        output = self.dense_3(x)
        # output = torch.sigmoid(self.dense_3(x))
        # output = torch.relu(self.dense_3(x))
        return output

model = CNN()
# optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
optimizer = optim.Adam(model.parameters())
criterion = nn.SmoothL1Loss()

def save(model, filename):
    save_filename = '{}.pt'.format(filename)
    torch.save(model, save_filename)
    print('Saved as %s' % save_filename)
    
def train(epoch):
    global lr
    model.train()
    batch_idx = 1
    total_loss = 0
    correct = 0
    for i in range(0, X_train.shape[0], config['batch_size']):
        if i + config['batch_size'] > X_train.shape[0]:
            x, y = X_train[i:], Y_train[i:]
        else:
            x, y = X_train[i:(i+config['batch_size'])], Y_train[i:(i+config['batch_size'])]
        if config['cuda']:
            x, y = Variable(torch.from_numpy(x).type(torch.FloatTensor), requires_grad=True).cuda(), Variable(torch.from_numpy(y)).cuda()
        else:
            x, y = Variable(torch.from_numpy(x).type(torch.FloatTensor), requires_grad=True), Variable(torch.from_numpy(y).type(torch.FloatTensor))
        # 将模型的参数梯度设置为0
        optimizer.zero_grad()
        output = model(x.unsqueeze(1))
        loss = criterion(output, y.view_as(output))
        # 后向传播调整参数
        loss.backward()
        # 根据梯度更新网络参数
        optimizer.step()
        batch_idx += 1
        # loss.item()能够得到张量中的元素值
        total_loss += loss.item()

    cur_loss = total_loss
    print('Train Epoch: {:2d}\t Learning rate: {:.4f}\t Loss: {:.6f} \n '.format(
                epoch, config['learning_rate'], cur_loss/batch_idx))
    
def evaluate(model):
    model.eval()
    batch_idx = 1
    total_loss = 0
    pred = np.array([])
    batch_size = config['batch_size']
    global min_mae
    for i in range(0, X_test.shape[0], batch_size):
        if i + batch_size > X_test.shape[0]:
            x, y = X_test[i:], Y_test[i:]
        else:
            x, y = X_test[i:(i+batch_size)], Y_test[i:(i+batch_size)]
        if False:
            x, y = Variable(torch.from_numpy(x).type(torch.FloatTensor), requires_grad=True).cuda(), Variable(torch.from_numpy(y)).cuda()
        else:
            x, y = Variable(torch.from_numpy(x).type(torch.FloatTensor), requires_grad=True), Variable(torch.from_numpy(y).type(torch.FloatTensor))
        with torch.no_grad():
            output = model(x.unsqueeze(1))
        loss = criterion(output, torch.tensor(y).view_as(output))
        pred = np.hstack((pred, output.flatten().numpy()))
        total_loss += loss.item()
    
    # print(Y_test, pred)
    mae = mean_absolute_error(Y_test, pred)
    rmse = np.sqrt(mean_squared_error(Y_test, pred))
    print('MAE: {}\t RMSE: {}\n'.format(mae, rmse))
    print('='*89)
    
    
    if mae < min_mae and mae < 9.41:
        min_mae = mae
        save(model, 'cnn_reg_avid_{:.2f}.pt'.format(mae))
    return total_loss

min_mae = 100

# for ep in range(1, config['epochs']):
#     train(ep)
#     tloss = evaluate(model)

model = torch.load('/Users/apple/Downloads/depression/cnn_reg_avid_9.30.pt.pt')
# model = BiLSTM(config)
# model.load_state_dict(lstm_model.state_dict())
evaluate(model)
    
