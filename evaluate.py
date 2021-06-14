#!/usr/bin/env python
# coding: utf-8

# In[5]:


import math
import torch
import numpy as np
from multietm import multiETM
from utils import get_topic_coherence, get_topic_diversity


## load data
data_path_I_test = './data/ICD/ICD_ehr_test.pt'
data_path_H_test = './data/HPO/HPO_ehr_test.pt'
data_path_Ip_test = './data/ICDp/ICDp_ehr_test.pt'
vocab_path_I = './data/ICD/ICD_vocab.txt'
vocab_path_H = './data/HPO/HPO_vocab.txt'
vocab_path_Ip = './data/ICDp/ICDp_vocab.txt'

ICD_input_test = torch.load(data_path_I_test)
HPO_input_test = torch.load(data_path_H_test)
ICDp_input_test =  torch.load(data_path_Ip_test)
num_docs_test = len(ICD_input_test)
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

train_tokens1 = np.load("./data/ICD/train_tokens1.npy",allow_pickle=True)
train_tokens2 = np.load("./data/HPO/train_tokens2.npy",allow_pickle=True)
train_tokens3 = np.load("./data/ICDp/train_tokens3.npy",allow_pickle=True)

## load model
model = multiETM(vocab1,vocab2,vocab3)
model = torch.load('./result/MultiEtm_500epochs_0.005lr_25topics.pkl')

## define evalute function
def evaluate(m_e, beta, eval_batch_size=1000, num_docs_test1 = num_docs_test, input_test1 = ICD_input_test, input_test21 = ICD_input_test, input_test22 = HPO_input_test, input_test23 = ICDp_input_test, input_test3 = HPO_input_test, input_test4 = ICDp_input_test, bow_norm = 1):
#     m.eval()
    with torch.no_grad():
        ## get \beta here
        beta = torch.from_numpy(beta) 
            
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
            data_batch_21 = input_test21[idx*eval_batch_size:(idx+1)*eval_batch_size]
            sums_21 = data_batch_21.sum(1).unsqueeze(1)
#             print('sums_21:',sums_21.shape)
            data_batch_22 = input_test22[idx*eval_batch_size:(idx+1)*eval_batch_size]
            sums_22 = data_batch_22.sum(1).unsqueeze(1)
#             print('sums_22:',sums_22.shape)
            data_batch_23 = input_test23[idx*eval_batch_size:(idx+1)*eval_batch_size]
            sums_23 = data_batch_23.sum(1).unsqueeze(1)
#             print('sums_23:',sums_23.shape)
            sums_2_tmp = sum((sums_21,sums_22),0)
            sums_2 = sum((sums_2_tmp,sums_23),0)
#             print('sums_2_shape',sums_2.shape)
            
            data_batch_2_tmp = torch.cat((data_batch_21,data_batch_22),1)
            data_batch_2 = torch.cat((data_batch_2_tmp,data_batch_23),1)
            
            res = torch.mm(theta, beta)
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
    ### 获得beta
    beta1 = model.model_i_decoder.get_beta()
    beta1 = beta1.data.numpy()
    beta1 = beta1.T
    print(beta1.shape)

    beta2 = model.model_h_decoder.get_beta()
    beta2 = beta2.data.numpy()
    beta2 = beta2.T
    print(beta2.shape)

    beta3 = model.model_ip_decoder.get_beta()
    beta3 = beta3.data.numpy()
    beta3 = beta3.T
    print(beta3.shape)
    beta_tmp = np.hstack((beta1,beta2))
    beta = np.hstack((beta_tmp,beta3))

    ### 获得 train_vocab
    train_tokens_all_list = []
    train_tokens_outlist1 = train_tokens1.tolist()
    train_tokens_outlist2 = train_tokens2.tolist()
    train_tokens_outlist3 = train_tokens3.tolist()
    for i in range(len(train_tokens_outlist1)):
        train_tokens_inlist1 = train_tokens_outlist1[i].tolist()
        train_tokens_all_list.append(train_tokens_inlist1)
    print(train_tokens_all_list[0])

    for i in range(len(train_tokens_outlist2)):
        train_tokens_inlist2 = train_tokens_outlist2[i].tolist()
        train_tokens_inlist_new2  = [i + 1102 for i in train_tokens_inlist2]
        train_tokens_all_list[i]=train_tokens_all_list[i]+train_tokens_inlist_new2
    print(train_tokens_all_list[0])

    for i in range(len(train_tokens_outlist3)):
        train_tokens_inlist3 = train_tokens_outlist3[i].tolist()
        train_tokens_inlist_new3  = [i + 2713 for i in train_tokens_inlist3]
        train_tokens_all_list[i]=train_tokens_all_list[i]+train_tokens_inlist_new3
    print(train_tokens_all_list[0])
    print(len(train_tokens_all_list))

    train_tokens_list = []
    for i in range(len(train_tokens_all_list)):
        train_tokens_list.append(np.array(train_tokens_all_list[i]))
    train_tokens = np.array(train_tokens_list)

    ### 获得 vocab
    vocab = vocab1 + vocab2 +vocab3

    TC = get_topic_coherence(beta, train_tokens, vocab)
    TD = get_topic_diversity(beta, 25) 
    print("TD:",TD)

    val_ppl = evaluate(model.model_encoder, beta, eval_batch_size=1000, num_docs_test1 = num_docs_test, input_test1 = ICD_input_test, input_test21 = ICD_input_test, input_test22 = HPO_input_test, input_test23 = ICDp_input_test, input_test3 = HPO_input_test, input_test4 = ICDp_input_test, bow_norm = 1)
    print('val_ppl',val_ppl)
    


# In[ ]:




