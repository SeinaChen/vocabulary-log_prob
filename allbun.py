import nltk
from nltk.corpus import brown

alltokens = brown.words(categories=brown.categories())
count = nltk.FreqDist([w.lower() for w in alltokens])
type_list=list(count.keys())
type_freq = list(count.values())
type_set = dict(zip(type_list,type_freq))

type_size = 20000
new_type_set = sorted(type_set.items(), key=lambda x:x[1], reverse=True)[:type_size]
type_dict = dict([(v[0],i) for i,v in enumerate(new_type_set)])
type_dict['<out of range>'] = len(type_dict)
type_dict['<unknow>'] = len(type_dict)
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

brown_token = brown.sents(categories=brown.categories())

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size *2* embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

# 文書の処理
#chat_bun = (' ').join(brown_token[0])
total_bun = []
for i in range(10):
    chat_bun = (' ').join(brown_token[i])
    process_idls = text_tokenize(chat_bun)
    total_bun += process_idls

CONTEXT_SIZE = 2 #3
EMBEDDING_DIM = 10 #2
fivgrams = [([process_idls[i], process_idls[i+1], process_idls[i+3], process_idls[i+4]], process_idls[i+2]) for i in range(len(process_idls) - 4)]
voca_size = len(type_dict)#1
losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(voca_size, EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)
    
for epoch in range(10):
    total_loss = 0
    for context, target in fivgrams:
        context_idxs = torch.tensor(context, dtype=torch.long)
        model.zero_grad()
        log_probs = model(context_idxs)
        loss = loss_function(log_probs, torch.tensor([target], dtype=torch.long))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    losses.append(total_loss)
print(losses) 
