num_embeddings = 50257
input_dim = 512
num_heads = 8
num_layers = 6
dropout = 0.0
tie = True

data_path = './'
batch_size = 128
mini_batch_size = 16
assert batch_size % mini_batch_size == 0
seq_len = 512

max_lr = 5e-4
wd = 0.2
betas = (0.9, 0.98)

train_steps = 50000
warmup_steps = 4000

exp_name = 'wikitext-103'