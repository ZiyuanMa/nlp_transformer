import math
import numpy as np
import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformer import Model
from data import load_data
import config


train_dataloader, valid_dataloader, _ = load_data(config.data_path, config.batch_size, config.mini_batch_size, config.seq_len)


original_model = Model(config.num_embeddings, config.input_dim, config.num_heads, config.num_layers, 
                        config.dropout, config.dropout, config.dropout, config.tie).cuda()
model = torch.compile(original_model)


num_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
print(f'number of param: {num_params}')
print(f'train batches: {len(train_dataloader)}')
print(f'valid batches: {len(valid_dataloader)}')


def group_weight(module):

    decay_group = []
    no_decay_group = []

    # no weight decay for norm and bias
    for n, p in module.named_parameters():
        if 'norm' in n or 'bias' in n:
            no_decay_group.append(p)
        else:
            decay_group.append(p)
        
    assert len(list(module.parameters())) == len(decay_group) + len(no_decay_group)
    groups = [dict(params=decay_group), dict(params=no_decay_group, weight_decay=.0)]

    return groups


weights = group_weight(model)
optimizer = torch.optim.AdamW(weights, lr=config.max_lr, weight_decay=config.wd, betas=config.betas)

lr_warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=config.warmup_steps)
lr_decay = CosineAnnealingLR(optimizer, config.train_steps-config.warmup_steps, eta_min=config.max_lr*0.01)
scheduler = SequentialLR(optimizer, [lr_warmup, lr_decay], [config.warmup_steps])


writer = SummaryWriter(f'./runs/{config.exp_name}')
scaler = torch.cuda.amp.GradScaler()
curr_step = 0
best_ppl = float('inf')
losses = []

while curr_step < config.train_steps:
    
    for data in tqdm(train_dataloader):

        data = data.cuda()
        
        if config.mini_batch_size > 0:
            
            num_mini_batches = config.batch_size // config.mini_batch_size
            batch_loss = 0

            for mini_data in data.chunk(num_mini_batches, 0):

                with torch.autocast(device_type="cuda"):
  
                    output = model(mini_data[:, :-1])
                    output = torch.reshape(output, (-1, output.size(-1)))
                    target = torch.reshape(mini_data[:, 1:], (-1,))
                    loss = F.cross_entropy(output, target) / num_mini_batches

                scaler.scale(loss).backward()
                batch_loss += loss.item()
            
        
        else:

            with torch.autocast(device_type="cuda"):
                
                output = model(data[:, :-1])
                output = torch.reshape(output, (-1, output.size(-1)))
                target = torch.reshape(data[:, 1:], (-1,))
                loss = F.cross_entropy(output, target)

            scaler.scale(loss).backward()
            batch_loss = loss.item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        scaler.step(optimizer)

        scaler.update()
        scheduler.step()
        optimizer.zero_grad()

        curr_step += 1

        writer.add_scalar('Perplexity/train_step', math.exp(batch_loss), curr_step)
        losses.append(batch_loss)

        if curr_step % 1000 == 0:
            print(f'train {curr_step}: {math.exp(sum(losses)/len(losses))}')
            writer.add_scalar('Perplexity/train', math.exp(sum(losses)/len(losses)), curr_step)

            losses.clear()
            model.eval()

            for data in tqdm(valid_dataloader):
                data = data.cuda()
                with torch.inference_mode(), torch.autocast(device_type="cuda"):

                    output = original_model(data[:, :-1])
                    output = torch.reshape(output, (-1, output.size(-1)))
                    target = torch.reshape(data[:, 1:], (-1,))
                    loss = F.cross_entropy(output, target, reduction='none')
        
                losses.append(loss.cpu().numpy())
            
            avg_ppl = math.exp(np.mean(np.concatenate(losses, 0)).item())
            print(f'valid {curr_step}: {avg_ppl}')

            if avg_ppl <= best_ppl:
                best_ppl = avg_ppl
                torch.save(original_model.state_dict(), f'./runs/{config.exp_name}/best_model.pt')
                
            writer.add_scalar('Perplexity/valid', avg_ppl, curr_step)

            losses.clear()
            model.train()
        if curr_step == config.train_steps:
            break