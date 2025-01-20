import torch
import torch.nn as nn
from encoder import TokenHandler, Encoder

input_ids= torch.tensor([
    [101, 7592, 2023, 2003, 1037, 2742, 102],  # [CLS] hello this is a test [SEP]
    [101, 2054, 2003, 1996, 2087, 2307, 102]   # [CLS] what is the best way [SEP]
])

text = "Tengo ganas de comer pollo"
handler=TokenHandler()
encoder=Encoder(token_handler=handler, n_blocks=2, head_count=4, model_hidden_size=100)
token_ids, position_ids=handler.process_text(text)
print(encoder(token_ids, position_ids).shape)