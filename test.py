import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
  
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

train = pd.read_csv('Airline Satisfaction/train.csv')
train = train.drop(['Unnamed: 0', 'id'], axis=1)
train['Gender'] = train['Gender'].map({'Male': 0, 'Female': 1})
train['Customer Type'] = train['Customer Type'].map({'disloyal Customer': 0, 'Loyal Customer': 1})
train['Type of Travel'] = train['Type of Travel'].map({'Personal Travel': 0, 'Business travel': 1})
train['Class'] = train['Class'].map({'Eco': 0, 'Eco Plus': 1, 'Business': 2})
train['satisfaction'] = train['satisfaction'].map({'neutral or dissatisfied': 0, 'satisfied': 1})
train = train.dropna()

test = pd.read_csv('Airline Satisfaction/test.csv')
test = test.drop(['Unnamed: 0', 'id'], axis=1)
test['Gender'] = test['Gender'].map({'Male': 0, 'Female': 1})
test['Customer Type'] = test['Customer Type'].map({'disloyal Customer': 0, 'Loyal Customer': 1})
test['Type of Travel'] = test['Type of Travel'].map({'Personal Travel': 0, 'Business travel': 1})
test['Class'] = test['Class'].map({'Eco': 0, 'Eco Plus': 1, 'Business': 2})
test['satisfaction'] = test['satisfaction'].map({'neutral or dissatisfied': 0, 'satisfied': 1})
test = test.dropna()

num_train_samples = len(train)
split_train_size = num_train_samples // 3

train_subset1 = train.iloc[:split_train_size]
train_subset2 = train.iloc[split_train_size:2*split_train_size]
train_subset3 = train.iloc[2*split_train_size:]

num_test_samples = len(test)
split_test_size = num_test_samples // 3

test_subset1 = test.iloc[:split_test_size]
test_subset2 = test.iloc[split_test_size:2*split_test_size]
test_subset3 = test.iloc[2*split_test_size:]

X=train_subset1.drop(['satisfaction'],axis=1)
y=train_subset1[['satisfaction']]

X_test=test_subset1.drop(['satisfaction'],axis=1)
y_test=test_subset1[['satisfaction']]

X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

class TrainData(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)

class TestData(Dataset):
    
    def __init__(self, X_data):
        self.X_data = X_data
        
    def __getitem__(self, index):
        return self.X_data[index]
        
    def __len__ (self):
        return len(self.X_data)

train_data = TrainData(torch.FloatTensor(X_train.values), 
                       torch.FloatTensor(y_train.values))

val_data = TrainData(torch.FloatTensor(X_val.values), 
                       torch.FloatTensor(y_val.values))

test_data = TestData(torch.FloatTensor(X_test.values))

EPOCHS = 20
BATCH_SIZE = 256
LEARNING_RATE = 0.001

train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=1)

class BinaryClassification(nn.Module):
    def __init__(self):
        super(BinaryClassification, self).__init__()
        # Number of input features is 12.
        self.layer_1 = nn.Linear(22, 120) 
        self.layer_2 = nn.Linear(120, 120)
        self.layer_3 = nn.Linear(120, 120)
        self.layer_4 = nn.Linear(120, 120)
        self.layer_5 = nn.Linear(120, 120)
        self.layer_6 = nn.Linear(120, 120)
        self.layer_7 = nn.Linear(120, 120)
        self.layer_out = nn.Linear(120, 1) 
        
        self.relu = nn.ReLU()
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.relu(self.layer_2(x))
        x = self.relu(self.layer_3(x))
        x = self.relu(self.layer_4(x))
        x = self.relu(self.layer_5(x))
        x = self.relu(self.layer_6(x))
        x = self.relu(self.layer_7(x))
        x = self.layer_out(x)
        
        return x
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = BinaryClassification()
model.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc

model.train()
for e in range(1, EPOCHS+1):
    epoch_loss = 0
    epoch_acc = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        
        y_pred = model(X_batch)
        
        loss = criterion(y_pred, y_batch)
        acc = binary_acc(y_pred, y_batch)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    model.eval()
    val_loss = 0
    val_acc = 0
    for X_val_batch, y_val_batch in val_loader:
        X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)

        y_val_pred = model(X_val_batch)

        val_batch_loss = criterion(y_val_pred, y_val_batch)
        val_batch_acc = binary_acc(y_val_pred, y_val_batch)

        val_loss += val_batch_loss.item()
        val_acc += val_batch_acc.item()

    # Calcul de la moyenne des métriques sur les batches
    epoch_loss /= len(train_loader)
    epoch_acc /= len(train_loader)
    val_loss /= len(val_loader)
    val_acc /= len(val_loader)

    with open("output.txt", "a") as f:
        f.write(f'Epoch {e:03}: | Train Loss: {epoch_loss:.5f} | Train Acc: {epoch_acc:.3f} | Val Loss: {val_loss:.5f} | Val Acc: {val_acc:.3f} \n')

y_pred_list = []
model.eval()
with torch.no_grad():
    for X_batch in test_loader:
        X_batch = X_batch.to(device)
        y_test_pred = model(X_batch)
        y_test_pred = torch.sigmoid(y_test_pred)
        y_pred_tag = torch.round(y_test_pred)
        y_pred_list.append(y_pred_tag.cpu().numpy())

y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

print(classification_report(y_test, y_pred_list))