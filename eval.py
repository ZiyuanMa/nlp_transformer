import math
import numpy as np
import torch
from torch.nn import functional as F

from tqdm import tqdm
from transformer import Model
from data import load_data
import config


_, valid_dataloader, test_dataloader = load_data(config.data_path, config.batch_size, config.mini_batch_size, config.seq_len)

model = Model(config.num_embeddings, config.input_dim, config.num_heads, config.num_layers, 
                        config.dropout, config.dropout, config.dropout, config.tie).cuda()
model.load_state_dict(torch.load('./best_model.pt'))

losses = []
model.eval()
for data in tqdm(valid_dataloader):
    data = data.cuda()
    with torch.inference_mode(), torch.autocast(device_type="cuda"):

        output = model(data[:, :-1])
        output = torch.reshape(output, (-1, output.size(-1)))
        target = torch.reshape(data[:, 1:], (-1,))
        loss = F.cross_entropy(output, target, reduction='none')

    losses.append(loss.cpu().numpy())

avg_ppl = math.exp(np.mean(np.concatenate(losses, 0)).item())
print(f'valid: {avg_ppl}')


losses.clear()
for data in tqdm(test_dataloader):
    data = data.cuda()
    with torch.inference_mode(), torch.autocast(device_type="cuda"):

        output = model(data[:, :-1])
        output = torch.reshape(output, (-1, output.size(-1)))
        target = torch.reshape(data[:, 1:], (-1,))
        loss = F.cross_entropy(output, target, reduction='none')

    losses.append(loss.cpu().numpy())

avg_ppl = math.exp(np.mean(np.concatenate(losses, 0)).item())
print(f'test: {avg_ppl}')

# 22.13 22.62