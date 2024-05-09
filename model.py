import numpy as np
import matplotlib.pyplot as plt
from pokerLoss import fourLoss

import torch
from torch import nn
import torch.nn.functional as F
import glob
from tqdm.auto import tqdm

from dataloader import *
import seaborn as sns

sns.set_theme()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate = 0.001

class GRUModel(nn.Module):
    def __init__(self, input_size, num_layers, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden_state = None
    
    def forward(self, x):
        output, self.hidden_state = self.gru(x, self.hidden_state,)
        output = self.fc(output)
        return output
    
    def init_hidden(self, batch_size):
        self.hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return self.hidden_state
    
# Input size for the baseline model: 15
# Input size with all features: 32
# Input size without the board: 22
model = GRUModel(input_size=15, num_layers=2, hidden_size=20, output_size=10).to(device)
print(f'Number Model Parameters: {sum(p.numel() for p in model.parameters())}')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()
    
train, test, val = load_batches(batch_size=256)

outer_training_loss_list = []
outer_validation_loss_list = []
outer_training_acc_list = []
outer_validation_acc_list = []

def accuracy(input_tensor, output_tensor):
    input_classes, output_classes = torch.argmax(input_tensor, dim=1), torch.argmax(output_tensor, dim=1)
    return (torch.count_nonzero(input_classes == output_classes)/input_classes.shape[0]).item()

for epoch in range(20):
    print(f'Starting Training for Epoch {epoch}')
    h = 0
    # for param in model.parameters():
    #     if(h < 5):
    #         print(param.data)
    #     h += 1
    model.train()
    loss_list = []
    acc_list = []
    for j in range(4):
        random_indices = np.random.permutation(range(len(train))).tolist()
        train_portion = [train[x] for x in [random_indices[i] for i in range((j*(len(train))//4), ((j+1)*len(train)//4))]]
        for i, input in enumerate(tqdm(train_portion)):
            target = input[:, 1:, -10:]
            input = input[:,:-1,:]
            hidden = model.init_hidden(input.shape[0])
            input, target, hidden = input.float().to(device), target.float().to(device), hidden.to(device)
            model.zero_grad()
            loss = 0
            for t in range(input.shape[1]):
                out = model(input[:,t,:].reshape(input.shape[0], 1, input.shape[2]))
                loss += criterion(out.view(out.shape[0], out.shape[2]), target[:,t,:])
                acc_list.append(accuracy(out.view(out.shape[0], out.shape[2]), target[:,t,:]))
            optimizer.zero_grad()
            loss_list.append((loss.item() / input.shape[1]))
            loss.backward()
            optimizer.step()
        outer_training_loss_list.append(np.mean(loss_list))
        outer_training_acc_list.append(np.mean(acc_list))
        print(f'Average Training Loss: {np.mean(loss_list)}')
        print(f'Average Training Accuracy: {outer_training_acc_list[-1]}')

        model.eval()
        with torch.no_grad():
            val_loss_list = []
            val_acc_list = []
            for i, input in enumerate(tqdm(val)):
                target = input[:, 1:, -10:]
                input = input[:,:-1,:]
                hidden = model.init_hidden(input.shape[0])
                input, target, hidden = input.float().to(device), target.float().to(device), hidden.to(device)
                loss = 0
                for t in range(input.shape[1]):
                    out = model(input[:,t,:].reshape(input.shape[0], 1, input.shape[2]))
                    loss += criterion(out.view(out.shape[0], out.shape[2]), target[:,t,:])
                    val_acc_list.append(accuracy(out.view(out.shape[0], out.shape[2]), target[:,t,:]))
                val_loss_list.append((loss.item() / input.shape[1]))
        outer_validation_loss_list.append(np.mean(val_loss_list))
        outer_validation_acc_list.append(np.mean(val_acc_list))
        print(f'Average Validation Loss: {np.mean(val_loss_list)}')
        print(f'Average Validation Accuracy: {outer_validation_acc_list[-1]}')

        if(len(outer_validation_acc_list) > 10):
            worsening_acc = []
            for i in range(1,5):
                worsening_acc.append(outer_validation_acc_list[-1] < outer_validation_acc_list[-(i+1)])
            if(all(worsening_acc)):
                break
    
plt.plot(outer_training_loss_list, label='training')
plt.plot(outer_validation_loss_list, label='validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Opponent Action Predictor Loss: More Features')
plt.savefig('model_loss.png')
plt.close()

plt.plot(outer_training_acc_list, label='training')
plt.plot(outer_validation_acc_list, label='validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Opponent Action Predictor Accuracy: More Features')
plt.savefig('model_accuracy.png')
plt.close()