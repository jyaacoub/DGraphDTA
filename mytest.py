#%%
import torch
from torchsummary import summary

data = 'davis'
path = f'/cluster/home/t122995uhn/projects/DGraphDTA/models/model_GNNNet_{data}.model'
m = torch.load(path, map_location=torch.device('cpu'))
# %%
print(m)
summary(m)
# %%
