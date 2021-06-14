#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch
import numpy as np
from torch import nn
import torch.nn.functional as F 
from transformers import AutoTokenizer, AutoModel

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
vocab_size1 = 1102
vocab_size2 = 1611
vocab_size3 = 1798
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###### Load Pretrained Biobert Model #####
bert_path = './pretrained_model/BioBERT/'
tokenizer = AutoTokenizer.from_pretrained(bert_path)
bert_model = AutoModel.from_pretrained(bert_path)

###### Inference Process ######
class ETM_Encoder(nn.Module):
    def __init__(self, num_topics=num_topics, vocab_size1=vocab_size1, vocab_size2=vocab_size2, vocab_size3=vocab_size3, t_hidden_size=t_hidden_size, theta_act=theta_act,enc_drop=enc_drop):
        super(ETM_Encoder, self).__init__()  
        ## define hyperparameters
        self.num_topics = num_topics
        self.vocab_size1 = vocab_size1 
        self.vocab_size2 = vocab_size2 
        self.vocab_size3 = vocab_size3 
        self.t_hidden_size = t_hidden_size
        self.theta_act = nn.ReLU(theta_act)
        self.enc_drop = enc_drop
        self.t_drop = nn.Dropout(enc_drop)
        
        
        ## define variational distribution for \theta_{1:D} via amortizartion
        self.q_theta1 = nn.Sequential(
                nn.Linear(vocab_size1, t_hidden_size), 
                self.theta_act,
                nn.Linear(t_hidden_size, t_hidden_size),
                self.theta_act,
            )
        self.q_theta2 = nn.Sequential(
                nn.Linear(vocab_size2, t_hidden_size), 
                self.theta_act,
                nn.Linear(t_hidden_size, t_hidden_size),
                self.theta_act,
            )
        self.q_theta3 = nn.Sequential(
                nn.Linear(vocab_size3, t_hidden_size), 
                self.theta_act,
                nn.Linear(t_hidden_size, t_hidden_size),
                self.theta_act,
            )
        self.mu_q_theta = nn.Linear(t_hidden_size*3, num_topics, bias=True)
        self.logsigma_q_theta = nn.Linear(t_hidden_size*3, num_topics, bias=True)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar) 
            eps = torch.randn_like(std)
            return eps.mul_(std).add_(mu)
        else:
            return mu

    def encode(self, bows1,bows2,bows3):
        q_theta1 = self.q_theta1(bows1)
        q_theta2 = self.q_theta2(bows2)
        q_theta3 = self.q_theta3(bows3)        
        
        if self.enc_drop > 0:
            q_theta1 = self.t_drop(q_theta1)
            q_theta2 = self.t_drop(q_theta2)
            q_theta3 = self.t_drop(q_theta3)
        q_theta12 = torch.cat((q_theta1, q_theta2), 1)
        q_theta = torch.cat((q_theta12, q_theta3), 1)
        
        mu_theta = self.mu_q_theta(q_theta)
        logsigma_theta = self.logsigma_q_theta(q_theta)
        kl_theta = - 0.5 * torch.sum(1 + logsigma_theta - mu_theta.pow(2) - logsigma_theta.exp(), dim=-1).mean()
        
        return mu_theta, logsigma_theta, kl_theta
    
    def get_theta(self, normalized_bows1, normalized_bows2, normalized_bows3):
        mu_theta, logsigma_theta, kld_theta = self.encode(normalized_bows1,normalized_bows2,normalized_bows3)
        z = self.reparameterize(mu_theta, logsigma_theta)
        theta = F.softmax(z, dim=1)
        
        return theta, kld_theta

    def forward(self, bows1, normalized_bows1, bows2, normalized_bows2, bows3, normalized_bows3, theta =None, aggregate=True):        
        ## get \theta
        if theta is None:
            theta, kld_theta = self.get_theta(normalized_bows1,normalized_bows2,normalized_bows3)
        else:
            kld_theta = None
        return theta, kld_theta

###### ICD9 diagnosis Generative Process ######
class ETM_ICD_Decoder(nn.Module):
    def __init__(self, vocab1, alpha, theta, kl_theta, emsize = emsize, num_topics=num_topics, vocab_size=vocab_size1, t_hidden_size=t_hidden_size, rho_size=rho_size, theta_act=theta_act, embeddings=None, train_embeddings=train_embeddings, enc_drop=enc_drop):
        super(ETM_ICD_Decoder, self).__init__()   
        ## define hyperparameters
        self.num_topics = num_topics
        self.vocab_size = vocab_size
        self.t_hidden_size = t_hidden_size
        self.rho_size = rho_size
        self.enc_drop = enc_drop
        self.emb_size = emsize
        self.t_drop = nn.Dropout(enc_drop)
        self.theta_act = nn.ReLU(theta_act)
        self.alphas = alpha 

        if not train_embeddings:
            vectors = {}             
            inputs = tokenizer(vocab1, return_tensors="pt", padding=True)
            outputs = bert_model(**inputs)
            embed_list = outputs.pooler_output.tolist()
            for i in range(len(embed_list)):
                line = embed_list[i]
                word = vocab1[i]
                vect = np.array(line).astype(np.float)
                vectors[word] = vect
            embeddings = np.zeros((self.vocab_size, self.emb_size))
            words_found = 0
            for i, word in enumerate(vocab1):
                try: 
                    embeddings[i] = vectors[word]
                    words_found += 1
                except KeyError:
                    embeddings[i] = np.random.normal(scale=0.6, size=(self.emb_size, ))
            embeddings = torch.from_numpy(embeddings).to(device)
            embeddings_dim = embeddings.size()
            print(embeddings.size())

        ## define the word embedding matrix \rho
        if train_embeddings:
             self.rho = nn.Parameter(torch.randn(self.rho_size, self.vocab_size))
        else:
            num_embeddings, emsize = embeddings.size()
            rho = nn.Embedding(num_embeddings, emsize)
            self.rho = embeddings.clone().float()   
    
    def get_activation(self, act):
        if act == 'tanh':
            act = nn.Tanh()
        elif act == 'relu':
            act = nn.ReLU()
        else:
            print('Defaulting to tanh activations...')
            act = nn.Tanh()
        return act 
    
    def get_beta(self):
        logit = torch.mm(self.alphas, self.rho.T) 
        beta = F.softmax(logit, dim=1).transpose(1, 0) ## softmax over vocab dimension
        return beta
    
    def decode(self, theta, beta):
        res = torch.mm(theta, beta)
        preds = torch.log(res+1e-6)
        return preds  
    
    def forward(self, bows, normalized_bows, theta, kl_theta, aggregate=True):        
        # get \beta
        beta = self.get_beta()
        ## get prediction loss
        preds = self.decode(theta, beta.T)
        recon_loss = -(preds * bows).sum(1)
        if aggregate:
            recon_loss = recon_loss.mean()

        return recon_loss, kl_theta, preds

###### HPO Generative Process ######
class ETM_HPO_Decoder(nn.Module):
    def __init__(self, vocab2, alpha, theta, kl_theta, emsize = emsize, num_topics=num_topics, vocab_size=vocab_size2, t_hidden_size=t_hidden_size, rho_size=rho_size, theta_act=theta_act, embeddings=None, train_embeddings=train_embeddings, enc_drop=enc_drop):
        super(ETM_HPO_Decoder, self).__init__()   
        ## define hyperparameters
        self.num_topics = num_topics
        self.vocab_size = vocab_size
        self.t_hidden_size = t_hidden_size
        self.rho_size = rho_size
        self.enc_drop = enc_drop
        self.emb_size = emsize
        self.t_drop = nn.Dropout(enc_drop)
        self.theta_act = nn.ReLU(theta_act)
        self.alphas = alpha 
#         print('alpha2:', self.alphas)

        if not train_embeddings:
            vectors = {}             
            inputs = tokenizer(vocab2, return_tensors="pt", padding=True)
            outputs = bert_model(**inputs)
            embed_list = outputs.pooler_output.tolist()
            for i in range(len(embed_list)):
                line = embed_list[i]
                word = vocab2[i]
                vect = np.array(line).astype(np.float)
                vectors[word] = vect
            embeddings = np.zeros((self.vocab_size, self.emb_size))
            words_found = 0
            for i, word in enumerate(vocab2):
                try: 
                    embeddings[i] = vectors[word]
                    words_found += 1
                except KeyError:
                    embeddings[i] = np.random.normal(scale=0.6, size=(self.emb_size, ))
            embeddings = torch.from_numpy(embeddings).to(device)
            embeddings_dim = embeddings.size()
            print(embeddings.size())
        
        ## define the word embedding matrix \rho
        if train_embeddings:
             self.rho = nn.Parameter(torch.randn(self.rho_size, self.vocab_size))
        else:
            num_embeddings, emsize = embeddings.size()
            rho = nn.Embedding(num_embeddings, emsize)
            self.rho = embeddings.clone().float()
#         print('rho2:', self.rho)
    
    def get_activation(self, act):
        if act == 'tanh':
            act = nn.Tanh()
        elif act == 'relu':
            act = nn.ReLU()
        else:
            print('Defaulting to tanh activations...')
            act = nn.Tanh()
        return act 
    def get_beta(self):
        logit = torch.mm(self.alphas, self.rho.T) #10*660
        beta = F.softmax(logit, dim=1).transpose(1, 0) ## softmax over vocab dimension
        return beta
    def decode(self, theta, beta):
        res = torch.mm(theta, beta.T)
        preds = torch.log(res+1e-6)
        return preds  
    def forward(self, bows, normalized_bows, theta, kl_theta, aggregate=True):        
        # get \beta
        beta = self.get_beta()
        
        ## get prediction loss
        preds = self.decode(theta, beta)
        recon_loss = -(preds * bows).sum(1)
        if aggregate:
            recon_loss = recon_loss.mean()
        return recon_loss, kl_theta, preds

    
###### Procedure Generative Process ######
class ETM_ICDP_Decoder(nn.Module):
    def __init__(self, vocab3, alpha, theta, kl_theta, emsize = emsize, num_topics=num_topics, vocab_size=vocab_size3, t_hidden_size=t_hidden_size, rho_size=rho_size, theta_act=theta_act, embeddings=None, train_embeddings=train_embeddings, enc_drop=enc_drop):
        super(ETM_ICDP_Decoder, self).__init__()   
        ## define hyperparameters
        self.num_topics = num_topics
        self.vocab_size = vocab_size
        self.t_hidden_size = t_hidden_size
        self.rho_size = rho_size
        self.enc_drop = enc_drop
        self.emb_size = emsize
        self.t_drop = nn.Dropout(enc_drop)
        self.theta_act = nn.ReLU(theta_act)
        self.alphas = alpha        

        if not train_embeddings:
            vectors = {}             
            inputs = tokenizer(vocab3, return_tensors="pt", padding=True)
            outputs = bert_model(**inputs)
            embed_list = outputs.pooler_output.tolist()
            for i in range(len(embed_list)):
                line = embed_list[i]
                word = vocab3[i]
                vect = np.array(line).astype(np.float)
                vectors[word] = vect
            embeddings = np.zeros((self.vocab_size, self.emb_size))
            words_found = 0
            for i, word in enumerate(vocab3):
                try: 
                    embeddings[i] = vectors[word]
                    words_found += 1
                except KeyError:
                    embeddings[i] = np.random.normal(scale=0.6, size=(self.emb_size, ))
            embeddings = torch.from_numpy(embeddings).to(device)
            embeddings_dim = embeddings.size()
            print(embeddings.size())
            
        ## define the word embedding matrix \rho
        if train_embeddings:
             self.rho = nn.Parameter(torch.randn(rho_size, vocab_size))
        else:
            num_embeddings, emsize = embeddings.size()
            rho = nn.Embedding(num_embeddings, emsize)
            self.rho = embeddings.clone().float()
        
    def get_activation(self, act):
        if act == 'tanh':
            act = nn.Tanh()
        elif act == 'relu':
            act = nn.ReLU()
        else:
            print('Defaulting to tanh activations...')
            act = nn.Tanh()
        return act 
    
    def get_beta(self):
        logit = torch.mm(self.alphas, self.rho.T) #10*660
        beta = F.softmax(logit, dim=1).transpose(1, 0) ## softmax over vocab dimension
        return beta
    
    def decode(self, theta, beta):
        res = torch.mm(theta, beta.T)
        preds = torch.log(res+1e-6)
        return preds  
    
    def forward(self, bows, normalized_bows, theta, kl_theta, aggregate=True):        
        # get \beta
        beta = self.get_beta()
        ## get prediction loss
        preds = self.decode(theta, beta)
        recon_loss = -(preds * bows).sum(1)
        if aggregate:
            recon_loss = recon_loss.mean()

        return recon_loss, kl_theta, preds

    
class multiETM(nn.Module):
    def __init__(self, vocab1, vocab2, vocab3):
        super(multiETM, self).__init__() 
        self.alphas = nn.Parameter(torch.randn(25, 768))        
        self.model_encoder = ETM_Encoder(num_topics=25, vocab_size1=vocab_size1, vocab_size2=vocab_size2, vocab_size3=vocab_size3, t_hidden_size=800, theta_act='relu',enc_drop=0.1)
        self.model_i_decoder = ETM_ICD_Decoder(vocab1 = vocab1, alpha = self.alphas, theta = None, kl_theta=None, emsize = 768, num_topics=25, vocab_size=vocab_size1, 
                                   t_hidden_size=800, rho_size=768, theta_act='relu', embeddings=None, 
                                   train_embeddings=0, enc_drop=0.1)        
        self.model_h_decoder = ETM_HPO_Decoder(vocab2 = vocab2, alpha = self.alphas, theta = None, kl_theta=None, emsize = 768, num_topics=25, vocab_size=vocab_size2, 
                                   t_hidden_size=800, rho_size=768, theta_act='relu', embeddings=None, 
                                   train_embeddings=0, enc_drop=0.1)
        self.model_ip_decoder = ETM_ICDP_Decoder(vocab3 = vocab3, alpha = self.alphas, theta = None, kl_theta=None, emsize = 768, num_topics=25, vocab_size=vocab_size3, 
                                   t_hidden_size=800, rho_size=768, theta_act='relu', embeddings=None, 
                                   train_embeddings=0, enc_drop=0.1)

    def forward(self, bows1, normalized_bows1, bows2, normalized_bows2, bows3, normalized_bows3):
        theta, kld_theta = self.model_encoder(bows1, normalized_bows1, bows2, normalized_bows2, bows3, normalized_bows3)
        
        recon_loss1, kld_theta, preds1 = self.model_i_decoder(bows1, normalized_bows1, theta = theta, kl_theta=kld_theta) 
        recon_loss2, kld_theta, preds2 = self.model_h_decoder(bows2, normalized_bows2, theta = theta, kl_theta=kld_theta)
        recon_loss3, kld_theta, preds3 = self.model_ip_decoder(bows3, normalized_bows3, theta = theta, kl_theta=kld_theta)

        return recon_loss1, recon_loss2, recon_loss3, kld_theta, self.model_i_decoder.get_beta(), self.model_h_decoder.get_beta(), self.model_ip_decoder.get_beta(), preds1, preds2, preds3, theta

# model = multiETM().to(device)
# print(model)


# In[ ]:




