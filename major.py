import os
import time
import copy
from datasets import load_dataset, load_from_disk
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns

from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def get_files_count(directory):
    return len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])

class CustomDataset(Dataset):
    def __init__(self, path_to_mode, tensor_shape = (1499,20)):
        self.path_to_eng = path_to_mode + "/eng/"
        self.path_to_arb = path_to_mode + "/arb/" 
        self.path_to_ger = path_to_mode + "/ger/" 
        
        self.eng_files_count = get_files_count(self.path_to_eng)
        self.arb_files_count = get_files_count(self.path_to_arb)
        self.ger_files_count = get_files_count(self.path_to_ger)
        self.tensor_shape = tensor_shape
           
    def __len__(self):
        return self.eng_files_count + self.arb_files_count + self.ger_files_count

    def __getitem__(self, idx):
        if idx < self.eng_files_count:
            file_index = idx
            # Specify the filename
            filename = self.path_to_eng + f"{file_index}.npy"
            # Load the array from the file
            mfcc = np.load(filename)
            label = 0
        elif (idx >= self.eng_files_count) and (idx < (self.eng_files_count+self.arb_files_count)):
            file_index = idx - self.eng_files_count
            # Specify the filename
            filename = self.path_to_arb + f"{file_index}.npy"
            # Load the array from the file
            mfcc = np.load(filename)
            label = 1
        elif (idx >= (self.eng_files_count+self.arb_files_count)) and (idx < (self.eng_files_count+self.arb_files_count+self.ger_files_count)):
            file_index = idx - (self.eng_files_count+self.arb_files_count)
            # Specify the filename
            filename = self.path_to_ger + f"{file_index}.npy"
            # Load the array from the file
            mfcc = np.load(filename)
            label = 2
        else:
             raise IndexError(f"Index {idx} is out of range for array of length {self.__len__()}")
         
        # Pad the array with zeros to match the predefined shape
        padded_array = np.pad(mfcc, [(0, self.tensor_shape[0] - mfcc.shape[0]), 
                                     (0, self.tensor_shape[1] - mfcc.shape[1])], 
                      mode='constant')
        # Convert the padded array to a PyTorch tensor
        tensor_from_array = torch.tensor(padded_array, dtype=torch.float32)
        label = torch.tensor(label)
        
        return tensor_from_array, label

batch_size = 100
shuffle = True
desired_shape = (437, 20)
data_root_path = "./mfccs2"

train_dataset = CustomDataset(data_root_path+"/train", desired_shape)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

val_dataset = CustomDataset(data_root_path+"/validation", desired_shape)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)

test_dataset = CustomDataset(data_root_path+"/test", desired_shape)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

#define a dictionary with the dataloaders
dataloaders ={"train":train_loader,
              "val":val_loader,
              "test":test_loader}

dataset_sizes = {'train':len(train_dataset),
                 'val': len(val_dataset),
                 "test":len(test_dataset)}

# Create an iterator from the DataLoader
train_iter = iter(dataloaders["train"])
# Get a batch of images without advancing the count
batch_mfccs, batch_labels = next(train_iter)
# Perform operations with the batch of images
print("Batch size: ", batch_mfccs.size(), "\nLables vector size: ", batch_labels.size())

print(dataset_sizes)

input_size = desired_shape[1]
sequence_length = desired_shape[0]

# Model Parameters
number_layers = 20
number_classes = 3

hidden_state_size = 64
bidirectional = False
drop_out = 0.5

class LSTM(nn.Module):
    def __init__(self,input_size, hidden_size ,number_layers, drop_out, bidirectional, number_classes):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.number_layers = number_layers
        
        self.drop_out = drop_out
        self.bidirectional = bidirectional
        self.number_classes = number_classes
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=number_layers, bidirectional=bidirectional, batch_first=True, dropout=drop_out)
        #                                                                                                                          x.shape -> (batch_size, sequence_length, input_size)   
        self.fc = nn.Linear(hidden_size*(2 if self.bidirectional else 1), number_classes)
        
    def forward(self,x):
        # create an initial hidden state, and cell state variables initiallized to zeros
        h0 = torch.zeros(self.number_layers*(2 if self.bidirectional else 1), x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.number_layers*(2 if self.bidirectional else 1), x.size(0), self.hidden_size).to(device)
        out , _ = self.lstm(x,(h0,c0))
        # out.shape -> (batch_size, sequence_length, hidden_size*(2 if self.bidirectional else 1))
        # we need to take the hidden state of only the last time step for classification
        out = out[:,-1,:]
        out = self.fc(out)
        return out

model = LSTM(input_size, hidden_state_size, number_layers, drop_out, bidirectional, number_classes).to(device)

writer = SummaryWriter("./LSTM/logs")
writer.add_graph(model, (batch_mfccs.to(device)))

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=100, print_frequency=100):
    print("*"*80, end='\n')
    val_error =[]
    train_error = []
    val_acc = []
    train_acc = []
    
    since = time.time() # ------------> to get the time taken by training----------------

    # ----------------- To save the weights of the model with the best accuracy----------
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    n_total_steps = len(dataloaders["train"])
    for epoch in range(num_epochs):
        if (epoch+1) % print_frequency == 0:
            print(f'Epoch {epoch+1}/{num_epochs} ===== Best accuracy reached: {best_acc*100:.4f}%')
            print('-' * 40)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            
            # 1. ------------------ Set the network mode---------------------------------
            """
                be aware that some layers have different behavior during train/evaluation
                (like BatchNorm, Dropout) so setting it matters.
            """
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            # ---------------------------------------------------------------------------

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for mfccs, labels in dataloaders[phase]:
                mfccs = mfccs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(mfccs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * mfccs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            # For learning rate scheduling
            # if phase == "train":
            #     scheduler.step(best_acc) 


            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if (epoch+1) % print_frequency == 0:
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc*100:.4f}'+"%")
                
            
            if phase =="train":
                train_error.append(epoch_loss)
                train_acc.append(epoch_acc)
                # writer.add_scalar("training loss"+setup_description, epoch_loss, epoch*n_total_steps)
                # writer.add_scalar("training accuracy"+setup_description, epoch_acc, epoch*n_total_steps)
            elif phase == "val":
                val_error.append(epoch_loss)
                val_acc.append(epoch_acc)
                # writer.add_scalar("validation loss"+setup_description, epoch_loss, epoch*n_total_steps)
                # writer.add_scalar("validation accuracy"+setup_description, epoch_acc, epoch*n_total_steps)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict()) # keep copy of the best model weights
        if (epoch+1) % print_frequency == 0:
            print("="*80, end='\n')   

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f"Best validation accuracy = {best_acc*100:4f} %")
    print(f"Training time {time_elapsed}")
    # writer.add_scalar("Best val Acc:"+setup_description, best_acc*100, epoch*n_total_steps)
    # writer.add_scalar("Training time"+setup_description, time_elapsed, epoch*n_total_steps)
    training_progress ={"train_acc":train_acc,
                        "train_err":train_error,
                        "val_acc":val_acc,
                        "val_err": val_error}

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, training_progress

# Training the model
epochs = 10
print_freq = 1
learning_rate = 0.001

# loss function
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
# optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0)
# lr_scheduler
lr_sched = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=15)


model, train_metrics = train_model(model, criterion, optimizer, lr_sched, dataloaders, dataset_sizes, epochs, print_freq)