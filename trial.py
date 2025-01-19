import torch
import torch.nn as nn
from transformers import AutoTokenizer

model_size=100
class TokenHandler():
    def __init__(self,):
        self.tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased")
        self.vocab_size=len(self.tokenizer.get_vocab())

    def process_text(self,text: str):
        tokens = self.tokenizer.tokenize(text)
        token_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(tokens))
        pos_ids = torch.tensor(list(range(len(token_ids))))
        return token_ids, pos_ids

    
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
    def __init__(self,head_size):
        super().__init__()
        self.query=nn.Linear()
        self.key=nn.Linear()
        self.value=nn.Linear()
        self.head_size=head_size
        self.dropout=nn.Dropout()

    def forward(self,input):
        query_tensor=self.query(input) #dim (B,T,C)
        key_tensor=self.key(input) #dim (B,T,C)
        value_tensor=self.value(input) #dim (B,T,C)
        weights_tensor=query_tensor@key_tensor.transpose(1,2)*self.head_size**-0.5 #(B, T, T)
        #weights_tensor=F.softmax(weights_tensor,dim=-1)
        weights_tensor=self.dropout(weights_tensor)
        output = weights_tensor@value_tensor# (B,T,T)*(B,T, C)->(B,T,C)
        return output

class MultiHead(nn.Module):
    def __init__(self,head_size,head_count):
        super().__init__()
        self.heads=[Head(head_size) for _ in range(head_count)]
        self.l1=nn.Linear()
        self.dropout=nn.Dropout()

    def forward(self, input):
        for head in self.heads:
            out = [head(input) for head in self.heads]
        output = self.dropout(self.l1(torch.concat(out, dim=-1)))
        #Now we concat a apply the next linear layer
        return output

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1=nn.Linear()
        self.l2=nn.Linear()
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout()

    def forward(self,input):
        output = self.dropout(self.l2(self.relu(self.l1(input))))
        return output
    
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.multihead=MultiHead()
        self.ff=FeedForward()
        self.layernorm1=nn.LayerNorm()
        self.layernorm2=nn.LayerNorm()
        #self.dropout()=nn.Dropout()

    def forward(self, input):
        x+= self.layernorm1(self.multihead(input))
        x+= self.layernorm2(self.ff(x))
        return x

class Encoder(nn.Module):
    def __init__(self,n_blocks):
        self.blocks=[Block() for _ in n_blocks]
        self.embedding_layer=Embedder()
    def forward(self,input):
        output = self.embedding_layer(input)
        for block in self.blocks:
            output = self.block(output)
        return output

class Model(nn.Module):
    def __init__(self):
        self.encoder = Encoder()
        self.output_layer = nn.Linear()

    def forward(self,input):
        encoder_output = self.encoder(input) #Un vector que codifica todos los tokens

print("Encoder Trial:")
text="mi perro muerde"
handle=TokenHandler()
token_id, pos_id =handle.process_text(text)
embd=Embedder(vocab_size=handle.vocab_size,model_hidden_size=model_size)
print(embd(input_token_id=token_id, input_position_id=pos_id).shape)
print(token_id.shape)
print(pos_id.shape)