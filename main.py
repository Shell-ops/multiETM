#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import torch
import pickle 
import numpy as np 
import os 
import math 
import random 
import sys
import matplotlib.pyplot as plt 
# import data
import scipy.io
import gensim
import ast

from multietm import multiETM
from torch import nn, optim
from torch.nn import functional as F


############################################# Parameters #############################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
### data parameters ###
data_path_I_train = './data/ICD/ICD_ehr_train.pt'
data_path_I_test = './data/ICD/ICD_ehr_test.pt'
data_path_H_train = './data/HPO/HPO_ehr_train.pt'
data_path_H_test = './data/HPO/HPO_ehr_test.pt'
data_path_Ip_train = './data/ICDp/ICDp_ehr_train.pt'
data_path_Ip_test = './data/ICDp/ICDp_ehr_test.pt'

vocab_path_I = './data/ICD/ICD_vocab.txt'
vocab_path_H = './data/HPO/HPO_vocab.txt'
vocab_path_Ip = './data/ICDp/ICDp_vocab.txt'

num_docs_train1 = 26022
num_docs_val1 = 1530
num_docs_test1 = 3060
bow_norm1 = 1

num_docs_train2 = 26022
num_docs_val2 = 1530
num_docs_test2 = 3060
bow_norm2 = 1

num_docs_train3 = 26022
num_docs_val3 = 1530
num_docs_test3 = 3060
bow_norm3 = 1

### model parameters ###
ckpt = './result/MultiEtm_500epochs_0.005lr_25topics.pkl'
clip = 0.0
wdecay = 1.2e-4
lr = 5e-3
batch_size = 1000
num_topics = 25
t_hidden_size=800
rho_size=768
emsize = 768 
theta_act='relu'
embeddings=None
train_embeddings=0
enc_drop=0.1
num_epochs = 500
eval_batch_size=1000


############################################# get data #############################################
#### 0. vocabulary
vocab1 = []
vocab2 = []
vocab3 = []
for line1 in open(vocab_path_I): 
    vocab1.append(line1[:-1])
for line2 in open(vocab_path_H): 
    vocab2.append(line2[:-1])
for line3 in open(vocab_path_Ip): 
    vocab3.append(line3[:-1])
vocab_size1 = len(vocab1)
vocab_size2 = len(vocab2)
vocab_size3 = len(vocab3)


#### 1. ICD9 Data
ICD_input_train = torch.load(data_path_I_train)[:26022]
ICD_input_val = torch.load(data_path_I_train)[26022:27552]       
ICD_input_test = torch.load(data_path_I_test)
train_tokens1 = np.load("./data/ICD/train_tokens1.npy",allow_pickle=True)

#### 2. HPO Data
HPO_input_train = torch.load(data_path_H_train)[:26022]
HPO_input_val = torch.load(data_path_H_train)[26022:27552]         
HPO_input_test = torch.load(data_path_H_test)
train_tokens2 = np.load("./data/HPO/train_tokens2.npy",allow_pickle=True)


##### 3. ICDp Data #####
ICDp_input_train = torch.load(data_path_Ip_train)[:26022]
ICDp_input_val = torch.load(data_path_Ip_train)[26022:27552]       
ICDp_input_test =  torch.load(data_path_Ip_test)
train_tokens3 = np.load("./data/ICDp/train_tokens3.npy",allow_pickle=True)


#############################################  Model and Optimizer ############################################# 
model = multiETM(vocab1,vocab2,vocab3).to(device)

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wdecay)

############################################# Trainer  ############################################# 
def train(epoch):
    recon_loss = []
    kl_loss = []
    model.train()
    acc_loss = 0
    acc_kl_theta_loss = 0
    acc_loss1 = 0
    acc_loss2 = 0
    acc_loss3 = 0
    cnt = 0
    ind = []
#     scheduler.step()

    for i in range(num_docs_train1):
        ind.append(int(i))
    indices = torch.tensor(ind)
#     indices = torch.randperm(num_docs_train)
    indices = torch.split(indices, batch_size)
    for idx, ind in enumerate(indices):
        optimizer.zero_grad()
        model.zero_grad()
        data_batch1 = ICD_input_train[idx*batch_size:(idx+1)*batch_size]
        data_batch2 = HPO_input_train[idx*batch_size:(idx+1)*batch_size]
        data_batch3 = ICDp_input_train[idx*batch_size:(idx+1)*batch_size]
        
        sums1 = data_batch1.sum(1).unsqueeze(1)
        sums2 = data_batch2.sum(1).unsqueeze(1)
        sums3 = data_batch3.sum(1).unsqueeze(1)
        sums3_list = sums3.tolist()
    
        if bow_norm1:
            normalized_data_batch1 = data_batch1 / sums1
        else:
            normalized_data_batch1 = data_batch1
        if bow_norm2:
            normalized_data_batch2 = data_batch2 / sums2
        else:
            normalized_data_batch2 = data_batch2
        if bow_norm3:
            normalized_data_batch3 = data_batch3 / sums3
        else:
            normalized_data_batch3 = data_batch3

        recon_loss1, recon_loss2, recon_loss3, kld_theta, beta1, beta2, beta3, preds1, preds2, preds3, theta= model(data_batch1, normalized_data_batch1, data_batch2, normalized_data_batch2, data_batch3, normalized_data_batch3)
        total_loss = recon_loss1 + recon_loss2 + recon_loss3 + kld_theta
        total_loss.backward()

        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        acc_loss += torch.sum(recon_loss1+recon_loss2+recon_loss3).item()
        acc_loss1 += torch.sum(recon_loss1).item()
        acc_loss2 += torch.sum(recon_loss2).item()
        acc_loss3 += torch.sum(recon_loss3).item()
        acc_kl_theta_loss += torch.sum(kld_theta).item()
        cnt += 1
    
    cur_loss = round(acc_loss / cnt, 2)
    cur_loss1 = round(acc_loss1 / cnt, 2) 
    cur_loss2 = round(acc_loss2 / cnt, 2) 
    cur_loss3 = round(acc_loss3 / cnt, 2) 
    cur_kl_theta = round(acc_kl_theta_loss / cnt, 2) 
    cur_real_loss = round(cur_loss + cur_kl_theta, 2)
    print('*'*100)
    print('Epoch----->{} .. LR: {} .. KL_theta: {} .. Rec_loss1: {} .. Rec_loss2: {} .. Rec_loss3: {} .. NELBO: {}'.format(
            epoch, optimizer.param_groups[0]['lr'], cur_kl_theta, cur_loss1, cur_loss2, cur_loss3, cur_real_loss))
    recon_loss.append(cur_loss)
    kl_loss.append(cur_kl_theta)
    print('*'*100)
    
    return cur_kl_theta, cur_loss, cur_real_loss

def evaluate(m_e, m_d, eval_batch_size=1000, num_docs_test1 = num_docs_val1, input_test1 = ICD_input_val, input_test2 = ICD_input_val, input_test3 = HPO_input_val, input_test4 = ICDp_input_val, bow_norm = 1):
#     m.eval()
    with torch.no_grad():
        ## get \beta here
        beta = m_d.get_beta()
        ### do dc and tc here
        acc_loss = 0
        cnt = 0
        indices_1 = torch.split(torch.tensor(range(num_docs_test1)), eval_batch_size)
        for idx, ind in enumerate(indices_1):
            ## get theta from first half of docs
            data_batch_1 = input_test1[idx*eval_batch_size:(idx+1)*eval_batch_size]
            sums_1 = data_batch_1.sum(1).unsqueeze(1)
            if bow_norm:
                normalized_data_batch_1 = data_batch_1 / sums_1
            else:
                normalized_data_batch_1 = data_batch_1
                
            data_batch_2 = input_test3[idx*eval_batch_size:(idx+1)*eval_batch_size]
            sums_2 = data_batch_2.sum(1).unsqueeze(1)
            if bow_norm:
                normalized_data_batch_2 = data_batch_2 / sums_2
            else:
                normalized_data_batch_2 = data_batch_2
                
            data_batch_3 = input_test4[idx*eval_batch_size:(idx+1)*eval_batch_size]
            sums_3 = data_batch_3.sum(1).unsqueeze(1)
            if bow_norm:
                normalized_data_batch_3 = data_batch_3 / sums_3
            else:
                normalized_data_batch_3 = data_batch_3
                
            theta, _ = m_e.get_theta(normalized_data_batch_1,normalized_data_batch_2,normalized_data_batch_3)
            
            ## get prediction loss using second half
            data_batch_2 = input_test2[idx*eval_batch_size:(idx+1)*eval_batch_size]
            sums_2 = data_batch_2.sum(1).unsqueeze(1)
            res = torch.mm(theta, beta.T)
            preds = torch.log(res)
            recon_loss = -(preds * data_batch_2).sum(1)
            
            loss = recon_loss / sums_2.squeeze()
            loss = loss.mean().item()
            acc_loss += loss
            cnt += 1
        cur_loss = acc_loss / cnt
        ppl_dc = round(math.exp(cur_loss), 1)
        return ppl_dc

    
if __name__ == "__main__":
    best_epoch = 0
    best_val_ppl = 1e9
    all_val_ppls = []
    kl_loss = []
    recon_loss = []
    all_loss = []
    for epoch in range(0, num_epochs):
        cur_kl_theta, cur_loss, cur_real_loss = train(epoch)
        kl_loss.append(cur_kl_theta)
        recon_loss.append(cur_loss)
        all_loss.append(cur_real_loss)         

        val_ppl_p = evaluate(model.model_encoder, model.model_i_decoder, eval_batch_size=eval_batch_size, num_docs_test1 = num_docs_val1, input_test1 = ICD_input_val, input_test2 = ICD_input_val, input_test3 = HPO_input_val, input_test4 = ICDp_input_val, bow_norm = bow_norm1)

        val_ppl_i = evaluate(model.model_encoder, model.model_h_decoder, eval_batch_size=eval_batch_size, num_docs_test1 = num_docs_val2, input_test1 = ICD_input_val, input_test2 = HPO_input_val, input_test3 = HPO_input_val, input_test4 = ICDp_input_val,  bow_norm = bow_norm2)

        val_ppl_h = evaluate(model.model_encoder, model.model_ip_decoder, eval_batch_size=eval_batch_size, num_docs_test1 = num_docs_val3, input_test1 = ICD_input_val, input_test2 = ICDp_input_val, input_test3 = HPO_input_val, input_test4 = ICDp_input_val,  bow_norm = bow_norm3)


        val_ppl = (val_ppl_p + val_ppl_i + val_ppl_h)/3.
        print('val_ppl', val_ppl)

        if val_ppl < best_val_ppl:
            with open(ckpt, 'wb') as f:
                torch.save(model, f)
            best_epoch = epoch
            best_val_ppl = val_ppl
        all_val_ppls.append(val_ppl)
    print('Training End')


# In[9]:


b = np.load("./data/HPO/train_tokens2.npy",allow_pickle=True)


# In[10]:


print(b.shape)


# In[ ]:




