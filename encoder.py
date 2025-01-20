import torch
import torch.nn as nn
from transformers import AutoTokenizer
from torch.nn import functional as F

class TokenHandler():
    def __init__(self):
        self.tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased")
        self.vocab_size=len(self.tokenizer.get_vocab())

    def process_text(self,text: str):
        tokens = self.tokenizer.tokenize(text)
        token_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(tokens))
        pos_ids = torch.tensor(list(range(len(token_ids))))
        return token_ids, pos_ids
    
    def mask_token_ids(self,token_ids,prob:float=0.15):
        mask_token_id = self.tokenizer.mask_token_id
        output = token_ids.clone()
        random_prob = torch.rand(size=token_ids.shape)
        print(random_prob)
        output[random_prob < prob*0.8] = mask_token_id
        count = int(random_prob[(random_prob>prob*0.8) & (random_prob<prob*0.9)].sum().item())
        random_token_ids = torch.randint(low=0,high=self.vocab_size, size= (count,))
        print(random_token_ids)
        output[(random_prob>prob*0.8) & (random_prob<prob*0.9)] = random_token_ids
        return output
    
class Embedder(nn.Module):
    def __init__(self, vocab_size: int, model_hidden_size: int):
        super().__init__()
        self.token_embd=nn.Embedding(vocab_size,model_hidden_size)
        self.pos_embd=nn.Embedding(vocab_size,model_hidden_size)
        #self.segment_embd=nn.Embedding(vocab_size,model_hidden_size)
        self.layer_norm= nn.LayerNorm(model_hidden_size,eps=1e-12)
        #self.dropout = nn.Dropout()
    def forward(self, 
                input_token_id=None,
                input_token_embd=None,  
                input_position_id=None)->torch.LongTensor:
        if input_token_id is None and input_token_embd is None:
            raise("Both input_token_id and input_token_embd cant be None")
        if input_token_embd is None: 
            token_encoding = self.token_embd(input_token_id)
        else:
            token_encoding=input_token_embd
        position_encoding=self.pos_embd(input_position_id)
        embeddings= token_encoding+position_encoding
        embeddings = self.layer_norm(embeddings)
        #embeddings = self.dropout(embeddings)
        return embeddings

class Head(nn.Module):
    def __init__(self,head_size,model_hidden_size):
        super().__init__()
        self.query=nn.Linear(model_hidden_size, head_size, bias=False)
        self.key=nn.Linear(model_hidden_size, head_size, bias=False)
        self.value=nn.Linear(model_hidden_size, head_size, bias=False)
        self.head_size=head_size
        #self.dropout=nn.Dropout()

    def forward(self,input):
        query_tensor=self.query(input) #dim (B,T,C)
        key_tensor=self.key(input) #dim (B,T,C)
        value_tensor=self.value(input) #dim (B,T,C)
        weights_tensor=query_tensor@key_tensor.transpose(-1,-2)*self.head_size**-0.5 #(B, T, T)
        weights_tensor=F.softmax(weights_tensor,dim=-1)
        #weights_tensor=self.dropout(weights_tensor)
        output = weights_tensor@value_tensor# (B,T,T)*(B,T, C)->(B,T,C)
        return output

class MultiHead(nn.Module):
    def __init__(self,head_count:int,model_hidden_size:int):
        super().__init__()
        self.heads=[Head(head_size=int(model_hidden_size/head_count),model_hidden_size=model_hidden_size) for _ in range(head_count)]
        self.l1=nn.Linear(model_hidden_size,model_hidden_size)
        #self.dropout=nn.Dropout()

    def forward(self, input):
        for head in self.heads:
            out = [head(input) for head in self.heads]
        output = self.l1(torch.concat(out, dim=-1))
        #Now we concat a apply the next linear layer
        return output

class FeedForward(nn.Module):
    def __init__(self,model_hidden_size:int):
        super().__init__()
        self.l1=nn.Linear(model_hidden_size,model_hidden_size)
        self.l2=nn.Linear(model_hidden_size,model_hidden_size)
        self.relu=nn.ReLU()
        #self.dropout=nn.Dropout()

    def forward(self,input):
        output = self.l2(self.relu(self.l1(input)))
        return output
    
class Block(nn.Module):
    def __init__(self,head_count:int, model_hidden_size:int ):
        super().__init__()
        self.multihead=MultiHead(head_count=head_count, model_hidden_size=model_hidden_size)
        self.ff=FeedForward(model_hidden_size=model_hidden_size)
        self.layernorm1=nn.LayerNorm(model_hidden_size, model_hidden_size)
        self.layernorm2=nn.LayerNorm(model_hidden_size, model_hidden_size)
        #self.dropout()=nn.Dropout()

    def forward(self, input):
        input += self.layernorm1(self.multihead(input))
        input += self.layernorm2(self.ff(input))
        return input

class Encoder(nn.Module):
    def __init__(self,n_blocks:int,head_count:int,model_hidden_size:int):
        super().__init__()
        self.blocks=[Block(head_count=head_count,model_hidden_size=model_hidden_size) for _ in range(n_blocks)]
        self.embedding_layer=Embedder(vocab_size=handle.vocab_size,model_hidden_size=model_hidden_size)
        self.mask_ids=mask_ids

    def forward(self,token_id,token_position_id):
        output = self.embedding_layer(input_token_id=token_id,input_position_id=token_position_id)
        for block in self.blocks:
            output = block(output)
        return output
"""
class Model(nn.Module):
    def __init__(self,n_blocks:int, head_count:int, model_hidden_size:int, final_layer):
        self.encoder = Encoder()
        self.final_layer = final_layer

    def forward(self,input):
        encoder_output = self.encoder(input) #Un vector que codifica todos los tokens
        output= self.final_layer(encoder_output)
        return output
        """
handler= TokenHandler()
text="Mi casa está sucia y tengo hambre"
token_ids,position_ids = handler.process_text(text)
print(token_ids)
print(handler.mask_token_ids(token_ids=token_ids))