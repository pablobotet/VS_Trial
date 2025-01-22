import torch
import torch.nn as nn
import torch.optim as optim
from encoder import TokenHandler, Model
import pandas as pd
import numpy as np
"""
input_ids= torch.tensor([0.5683, 0.2444, 0.8214, 0.7121, 0.9011, 0.4472, 0.6314, 0.3203, 0.4827,
        0.7146, 0.6750, 0.2651, 0.5054, 0.0414, 0.3709, 0.7045, 0.6106, 0.2312,
        0.5699, 0.8359, 0.5975, 0.1985, 0.6537, 0.1564, 0.7221, 0.1255, 0.8803,
        0.7323, 0.5261, 0.0404, 0.5466, 0.4842, 0.0282, 0.4162, 0.8259, 0.6732,
        0.9155, 0.7138, 0.5880, 0.2546, 0.1455, 0.8576, 0.9878, 0.2258, 0.1466,
        0.6811, 0.7486, 0.0063, 0.9750, 0.1726, 0.0064]
        )
prob=0.15
mask_cond = input_ids <prob 
print(mask_cond[(input_ids>prob*0.8) & (input_ids<prob*0.9)].sum().item())
"""

filepath = "/Users/user/Downloads/archive/combined_emotion.csv"

data_df = pd.read_csv(filepath_or_buffer=filepath)

text_np=np.array(data_df['sentence'])
#text = text_df[1]
handler=TokenHandler()
model=Model(vocab_size=handler.vocab_size,n_blocks=2,head_count=4, model_hidden_size=100)
def crossentropy(masked_output, labels):
    """
    masked_output-> the output of the encoder from a masked token_id sequence

    label-> a tensor with token_ids of the original sentence. -100 represents that a token has not been changed, and therefore should be ignored for the calculations
    """
    loss_fn=nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fn(masked_output,labels)
    return loss
print(model.parameters())
for text in text_np:
    optimizer=optim.AdamW(params=model.parameters(),lr=1e-5)
    loss_fn=nn.CrossEntropyLoss(ignore_index=-100)
    token_ids,position_ids = handler.process_text(text)
    masked_ids,labels =handler.mask_token_ids(token_ids)
    model.train()
    optimizer.zero_grad()
    logits =model(masked_ids,position_ids)
    loss=loss_fn(logits, labels)
    loss.backward()
    optimizer.step()
    print(loss)
