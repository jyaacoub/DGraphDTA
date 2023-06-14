#%%
import json
yourdir = "davis/"
# 1 fold = 1/6 of entire database
test_fold = json.load(open(yourdir + "folds/test_fold_setting1.txt"))
train_folds = json.load(open(yourdir + "folds/train_fold_setting1.txt")) # 5 folds
#%%
import pickle
import numpy as np


Y = pickle.load(open(yourdir+"Y", "rb"), encoding='latin1')
print(Y.shape) # (d,p) Davis has 68 drugs and 442 proteins
label_row_inds, label_col_inds = np.where(np.isnan(Y)==False)

#%% getting the row and column for the drug and protein in the test fold
test_drug_i = label_row_inds[test_fold]
test_prot_i= label_col_inds[test_fold]
# %%
