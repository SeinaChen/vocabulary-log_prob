######### 単語頻度順辞書 #########
import nltk
from nltk.corpus import brown
import numpy as np

#len(allwords) = 1161192
alltokens = brown.words(categories=brown.categories())

#word and frequency
#len(count) = 49815

count = nltk.FreqDist([w.lower() for w in alltokens])

type_list=list(count.keys())
type_freq = list(count.values())
type_set = dict(zip(type_list,type_freq))

#control size of word
type_size = 2000
new_type_set = sorted(type_set.items(), key=lambda x:x[1], reverse=True)[:type_size]
type_dict = dict([(v[0],i) for i,v in enumerate(new_type_set)])

#<out of range> ID = 2000
type_dict['<out of range>'] = len(type_dict)
#<unknow>ID = 2001
type_dict['<unknow>'] = len(type_dict)


#word => ID
def text_tokenize(chat_bun):
    tokens = nltk.word_tokenize(chat_bun)
    tokens[0:0]= ['<out of range>','<out of range>']
    tokens += ['<out of range>','<out of range>']
    tokens_lower_list = [i.lower() for i in tokens]
    id_ls = []
    for i in tokens_lower_list:
        if i in type_dict:
            id_ls.append(type_dict[i])
        else:
            i = "<unknow>"
            id_ls.append(type_dict[i])
    return  id_ls


######### 空白modelを定義する #########
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size *2* embedding_dim, 600)
        self.linear2 = nn.Linear(600, vocab_size, bias = False)
        self.linear3 = nn.Linear(context_size *2* embedding_dim, vocab_size)
        
    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out1 = self.linear2(out)
        out2 = F.relu(self.linear3(embeds))
        out = out1 + out2
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


######### model ロード ##########
model = torch.load("brown_model")

######## chat_bun 入力 ##########
chat_bun = ["The Fulton County Grand Jury said Friday an investigation of Atlanta's recent primary election produced `` no evidence '' that any irregularities took place ."]
chat_bun_idls = text_tokenize(chat_bun[0])
print(chat_bun_idls)


for sent in chat_bun_idls:
    for t in range(2, len(chat_bun_idls) - 2):
        context = chat_bun_idls[t-2:t] + chat_bun_idls[t+1:t+1+2]
        print(context)
        
        target = chat_bun_idls[t]
        print(target)
        
        chat_context_idxs = torch.tensor(context, dtype=torch.long)
        

        log_probs = model(chat_context_idxs)
 
        likelihood = np.exp(log_probs.data.numpy()[0][target])   

        keys = [k for k, v in type_dict.items() if v == target]
        print('IDは{}-単語{}の確率は:{}'.format(target,keys,likelihood),"\n") 
