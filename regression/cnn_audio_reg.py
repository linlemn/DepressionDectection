import numpy as np
import pandas as pd
import wave
import librosa
from python_speech_features import *

import torch 
from torch.utils import data # 获取迭代数据
from torch.autograd import Variable # 获取变量
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.nn import functional as F

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

prefix = '/Users/apple/Downloads/depression/'

train_split_df = pd.read_csv(prefix+'train_split_Depression_AVEC2017 (1).csv')
test_split_df = pd.read_csv(prefix+'dev_split_Depression_AVEC2017.csv')
train_split_num = train_split_df[['Participant_ID']]['Participant_ID'].tolist()
test_split_num = test_split_df[['Participant_ID']]['Participant_ID'].tolist()
train_split_clabel = train_split_df[['PHQ8_Score']]['PHQ8_Score'].tolist()
test_split_clabel = test_split_df[['PHQ8_Score']]['PHQ8_Score'].tolist()

def extract_features(number, audio_features, target, audio_targets, mode):
    transcript = pd.read_csv(prefix+'{0}_P/{0}_TRANSCRIPT.csv'.format(number), sep='\t').fillna('')
    
    wavefile = wave.open(prefix+'{0}_P/{0}_AUDIO.wav'.format(number, 'r'))
    sr = wavefile.getframerate()
    nframes = wavefile.getnframes()
    wave_data = np.frombuffer(wavefile.readframes(nframes), dtype=np.short)
    
    time_range = []
    response = ''
    response_flag = False
    time_collect_flag = False
    start_time = 0
    stop_time = 0

    signal = []
    
    global counter_train

    for t in transcript.itertuples():
        # participant一句话结束
        if getattr(t,'speaker') == 'Ellie':
            continue
        elif getattr(t,'speaker') == 'Participant':
            if 'scrubbed_entry' in getattr(t,'value'):
                continue
            start_time = int(getattr(t,'start_time')*sr)
            stop_time = int(getattr(t,'stop_time')*sr)
            signal = np.hstack((signal, wave_data[start_time:stop_time].astype(np.float)))
        
    # 1分钟
    clip = sr*1*15
    if target >= 10 and mode == 'train':
        times = 3 if counter_train < 48 else 2
        for i in range(times):
            if clip*(i+1) > len(signal):
                continue
            melspec = librosa.feature.melspectrogram(signal[clip*i:clip*(i+1)], n_mels=80,sr=sr)
#             melspec = base.logfbank(signal[clip*i:clip*(i+1)], samplerate=sr, winlen=0.064, winstep=0.032, nfilt=80, nfft=1024, lowfreq=130, highfreq=6854)
            logspec = melspec 
            audio_features.append(logspec)
            audio_targets.append(target)
            counter_train+=1
    else:  
        melspec = librosa.feature.melspectrogram(signal[:clip], n_mels=80, sr=sr)
#         melspec = base.logfbank(signal[:clip], samplerate=sr, winlen=0.064, winstep=0.032, nfilt=80, nfft=1024, lowfreq=130, highfreq=6854)
        logspec = melspec 
        audio_features.append(logspec) 
        audio_targets.append(target)
#     print(melspec.shape)
    print('{}_P feature done'.format(number))
    
# training set
audio_features_train = []
audio_ctargets_train = []

# test set
audio_features_test = []
audio_ctargets_test = []
mark_test = []

counter_train = 0
counter_test = 0

# training set
for index in range(len(train_split_num)):
    extract_features(train_split_num[index], audio_features_train, train_split_clabel[index], audio_ctargets_train, 'train')

# test set
for index in range(len(test_split_num)):
    extract_features(test_split_num[index], audio_features_test, test_split_clabel[index], audio_ctargets_test, 'test')

print(np.shape(audio_ctargets_train), np.shape(audio_ctargets_test))
print(counter_train, counter_test)

print("Saving npz file locally...")
np.savez(prefix+'data/audio/train_samples_reg.npz', audio_features_train)
np.savez(prefix+'data/audio/train_labels_reg.npz', audio_ctargets_train)
np.savez(prefix+'data/audio/test_samples_reg.npz', audio_features_test)
np.savez(prefix+'data/audio/test_labels_reg.npz', audio_ctargets_test)

config = {
    'num_classes': 1,
    'dropout': 0.5,
    'rnn_layers': 2,
    'embedding_size': 80,
    'batch_size': 2,
    'epochs': 30,
    'learning_rate': 1e-4,
    'cuda': True,
}

X_train = np.array(audio_features_train)
Y_train = np.array(audio_ctargets_train)
X_test = np.array(audio_features_test)
Y_test = np.array(audio_ctargets_test)

optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)

def save(model, filename):
    save_filename = '{}.pt'.format(filename)
    torch.save(model, save_filename)
    print('Saved as %s' % save_filename)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv2d_1 = nn.Conv2d(1, 32, (1,7), 1)
        self.conv2d_2 = nn.Conv2d(32, 32, (1,7), 1)
        self.dense_1 = nn.Linear(120736, 128)
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
        x = x.contiguous().view(-1, 120736)
        x = F.relu(self.dense_1(x))
        x = F.relu(self.dense_2(x))
        x = self.dropout(x)
        # output = torch.sigmoid(self.dense_3(x))
        output = torch.relu(self.dense_3(x))
        return output
    
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
            x, y = Variable(torch.from_numpy(x).type(torch.FloatTensor), requires_grad=True), Variable(torch.from_numpy(y))
        # 将模型的参数梯度设置为0
        optimizer.zero_grad()
        output = model(x.unsqueeze(1))
        loss = criterion(output, y)
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
        loss = criterion(output, torch.tensor(y))
        pred = np.hstack((pred, output.flatten().numpy()))
        total_loss += loss.item()
    
    print(Y_test, pred)
    print('MAE: {}'.format(mean_absolute_error(Y_test, pred)))
    print('='*89)

    return total_loss

model = CNN()
criterion = nn.CrossEntropyLoss()

for ep in range(1, config['epochs']):
    train(ep)
    tloss = evaluate()