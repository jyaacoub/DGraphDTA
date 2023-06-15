#%%
import torch
from torchsummary import summary

data = 'davis'
path = f'models/model_GNNNet_{data}.model'
m = torch.load(path, map_location=torch.device('cpu'))
# %%
print(m)
summary(m)
# %%
for key in list(m.keys()):
    print(key)
