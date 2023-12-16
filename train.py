import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 245 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-3 # self attention can't handle high learning rates, now lower because neural network is bigger
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384 # n_emd = 384, n_head = 6, every head is (384/6) 64 dimensional
n_head = 6
n_layer = 6
dropout = 0.2 # randomly prevent values from propogating
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# query - what am i looking for
# key - what do I contain
# value - what is the value


# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.positional_embedding_table = nn.Embedding(block_size, n_embd)

        # self.ffwd = FeedFoward(n_embd) # to research
        
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)


    def forward(self, idx, targets=None):
        # idx = input tensor
        # targets = target
        # token embedding table - contains relation between words

        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.positional_embedding_table(torch.arange(T, device=device)) # (T, C)

        x = tok_emb + pos_emb # x holds token identities and where they occur
        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.lm_head(x) # decoder (B, T, vocab_size)

        # produces (B, T, C), where:
        # B - batch size - how many examples you look at before making a weight update during one forward and backward pass and updatign weights
        # T - block size, context length
        # C - embedding. every token in the input is turned into an embedding

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) # cross entropy expects (input, target, weight)
                                                    # input - logits - batch size x sequence length
                                                    # target - logits - batch size x sequence length
                                                    # weight - current weights (embeddings)
        return logits, loss
            
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):

            # [:.. gets all rows/batches -block_size, gets last block
            idx_cond = idx[:, -block_size:]
            
            # invokes forward function of current object
            logits, loss = self(idx_cond)
            
            # [:... get all rows/batches, select last timestep, ..:] gets classes (tokens of vocabulary too)
            logits = logits[:, -1, :] # becomes (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)

            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

        return idx

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd) # research, this is the projection
        self.dropout = nn.Dropout(dropout) # randomly shuts off subset of neurons

    def forward(self, x):
        out =  torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """


    def __init__(self, n_embd):
        super().__init__()

        # why * 4? 
        # position-wise feed forward network
        # While the linear transformations are the smae across different positions, they use differen parmameters
        # layer to layer. Another way of describing this is as two convolutions with kernel size 1.
        # The dimensionality of input and output is d_model = 512, and the inner layer has dimensionality 2048. so we multiply by 4.

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # projection
            nn.Dropout(dropout) # randomly drops out
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ tranformer block: communication followed by computation """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head

        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class Head(nn.Module):
    
    """ one head of self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # tril is not a parameter of model, so we use buffer, and have to assign it in module
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout) # sometimes automatically drops out

    def forward(self, x):
        B, T, C = x.shape
        # produces (B, T, C), where:
        # B - batch size - how many examples you look at before making a weight update during one forward and backward pass and updatign weights
        # T - block size, context length
        # C - embedding. every token in the input is turned into an embedding

        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)
        # compute self attention scores
        # normalise it, scaled attention
        wei = q @ k.transpose(-2, -1) * C **-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        # decoder block, because can't communicate with the past
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # prevents communication with the past, (B, T, T)
        wei = F.softmax(wei, dim =- 1) # (B, T, T)
        wei = self.dropout(wei) # sometimes automatically drops out
        # perform the weighted aggregation of the values
        v = self.value(x) # (B, T< C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

model = BigramLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
