#%%
import torch
from torchsummary import summary
from gnn import GNNNet

data = 'kiba'
model_file_name = f'models/model_GNNNet_{data}.model'
model_st = GNNNet.__name__
cuda_name = 'cuda:0'
device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')
print('cuda_name:', cuda_name)
print('device:', device)

# %%
model = GNNNet()
model.to(device)
 
cp = torch.load(model_file_name, map_location=device) # loading checkpoint

# %% renaming keys to be compatible with torch 2.0
broken_keys = ['mol_conv1.weight', 'mol_conv2.weight', 'mol_conv3.weight', 'pro_conv1.weight', 'pro_conv2.weight', 'pro_conv3.weight']
for key in broken_keys:
    new_key = key.replace('.weight', '.lin.weight')
    # transpose weights to be compatible with torch 2.0
    cp[new_key] = cp.pop(key).transpose(0, 1)   

model.load_state_dict(cp)

# %% Calc model size
param_size = 0
for param in model.parameters():
    param_size += param.nelement() * param.element_size()
buffer_size = 0
for buffer in model.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

size_all_mb = (param_size + buffer_size) / 1024**2
print('model size: {:.3f}MB'.format(size_all_mb))

# %% Saving as new model
save_path = f'models/model_GNN_{data}_t2.model'
# torch.save(model.state_dict(), save_path)
# %%
