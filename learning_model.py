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

#word list
type_list_up = type_dict.keys()

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

#確率の計算
CONTEXT_SIZE = 2 #3
EMBEDDING_DIM = 100 #2


#辞書のサイズ 
voca_size = len(type_dict)#1
print("単語typeのサイズ：", voca_size,"\n")

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

losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(voca_size, EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)


for epoch in range(10):
    total_loss = 0
    brown_ids = [' '.join(s) for s in brown.sents(categories=brown.categories())][0:2]
    for sent in brown_ids:
        #print("文の分割",sent)
        process_idls = text_tokenize(sent)
        #print("文書ID：",process_idls)
        for t in range(2, len(process_idls) - 2):
            context = process_idls[t-2:t] + process_idls[t+1:t+1+2]
            #print("context",context)
            target = process_idls[t]
            #print("target",target)

            context_idxs = torch.tensor(context, dtype=torch.long)

            model.zero_grad()

            log_probs = model(context_idxs)
            
            loss = loss_function(log_probs, torch.tensor([target], dtype=torch.long))
        
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        losses.append(total_loss)
print(losses) 
    
#torch.save(model.state_dict(), 'brown_model')   
torch.save(model, 'brown_model')   
