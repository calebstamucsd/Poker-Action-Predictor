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
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
criterion = torch.nn.MSELoss()
    
train, train_targets, test, test_targets, val, val_targets = load_batches_equity(batch_size=128)

outer_training_loss_list = []
outer_validation_loss_list = []

def accuracy(input_tensor, output_tensor):
    input_classes, output_classes = torch.argmax(input_tensor, dim=1), torch.argmax(output_tensor, dim=1)
    return (torch.count_nonzero(input_classes == output_classes)/input_classes.shape[0]).item()

for epoch in range(6):
    print(f'Starting Training for Epoch {epoch}')
    h = 0
    model.train()
    loss_list = []
    for j in range(4):
        random_indices = np.random.permutation(range(len(train))).tolist()
        train_portion = [train[x] for x in [random_indices[i] for i in range((j*(len(train))//4), ((j+1)*len(train)//4))]]
        target_portion = [train_targets[x] for x in [random_indices[i] for i in range((j*(len(train))//4), ((j+1)*len(train)//4))]]
        for i, (input, target) in enumerate(tqdm(zip(train_portion, target_portion))):
            hidden = model.init_hidden(input.shape[0])
            input, target, hidden = input.float().to(device), target.float().to(device), hidden.to(device)
            model.zero_grad()
            for t in range(input.shape[1]):
                out = model(input[:,t,:].reshape(input.shape[0], 1, input.shape[2]))
            loss = criterion(out, target)
            optimizer.zero_grad()
            loss_list.append((loss.item() / input.shape[1]))
            loss.backward()
            optimizer.step()
        outer_training_loss_list.append(np.mean(loss_list))
        print(f'Average Training Loss: {np.mean(loss_list)}')

        model.eval()
        with torch.no_grad():
            val_loss_list = []
            for i, (input, target) in enumerate(tqdm(zip(val, val_targets))):
                hidden = model.init_hidden(input.shape[0])
                input, target, hidden = input.float().to(device), target.float().to(device), hidden.to(device)
                for t in range(input.shape[1]):
                    out = model(input[:,t,:].reshape(input.shape[0], 1, input.shape[2]))
                loss = criterion(out, target)
                val_loss_list.append((loss.item() / input.shape[1]))
        outer_validation_loss_list.append(np.mean(val_loss_list))
        print(f'Average Validation Loss: {np.mean(val_loss_list)}')
    
plt.plot(outer_training_loss_list, label='training')
plt.plot(outer_validation_loss_list, label='validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Opponent Action Predictor Loss: More Features')
plt.savefig('model_loss.png')
plt.close()