import torch
import torch.nn as nn
from transformers import AutoTokenizer

input_ids= torch.tensor([
    [101, 7592, 2023, 2003, 1037, 2742, 102],  # [CLS] hello this is a test [SEP]
    [101, 2054, 2003, 1996, 2087, 2307, 102]   # [CLS] what is the best way [SEP]
])

probability_tensor = torch.rand(size=input_ids.shape,requires_grad=False)
masked_input_ids=input_ids.clone()

mask_token_id = 103

def maskable_id(id:int,null_set={101,102})->bool:
    return id not in null_set

masked_condition = probability_tensor<0.15
masked_input_ids[masked_condition] = mask_token_id
print(masked_condition)
print(masked_input_ids)
print(mask_token_id)