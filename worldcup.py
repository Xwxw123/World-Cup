import csv
import math
import random
import torch
import torch.nn.functional as F

random.seed(42)
torch.manual_seed(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device: {device}")

# ----------------------------
# Load FIFA data
# ----------------------------

rows = []
with open('fifa/results.csv', newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        if row['home_score'] == 'NA' or row['away_score'] == 'NA':
            continue
        if int(row['date'][:4]) <= 1992:
            continue
        rows.append(row)

rows.sort(key=lambda r: r['date'])
min_year = int(rows[0]['date'][:4])
max_year = int(rows[-1]['date'][:4])
print(f"matches loaded: {len(rows)}  ({min_year} - {max_year})")

# ----------------------------
# Load FIFA rankings
# ----------------------------

rank_snapshots = {}  # (year, semester) -> {team: rank}
with open('fifa/fifa_mens_rank.csv', newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        key = (int(row['date']), int(row['semester']))
        rank_snapshots.setdefault(key, {})[row['team']] = int(row['rank'])

def get_rank(team, date_str):
    year  = int(date_str[:4])
    month = int(date_str[5:7])
    sem   = 1 if month <= 6 else 2
    for y, s in [(year, sem), (year, 3 - sem), (year - 1, 2), (year - 1, 1)]:
        snap = rank_snapshots.get((y, s), {})
        if team in snap:
            return snap[team]
    return None

def rank_bucket(rank):
    if rank is None:       return 'rank_unk'
    if rank <= 10:         return 'rank_top10'
    if rank <= 25:         return 'rank_top25'
    if rank <= 50:         return 'rank_top50'
    if rank <= 100:        return 'rank_top100'
    if rank <= 200:        return 'rank_top200'
    return 'rank_200plus'

RANK_TOKENS = ['rank_top10', 'rank_top25', 'rank_top50', 'rank_top100', 'rank_top200', 'rank_200plus', 'rank_unk']
print(f"rankings loaded: {len(rank_snapshots)} snapshots")

# ----------------------------
# Vocabulary: one token per team, one token per score character, separator, BOS
# ----------------------------

teams       = sorted(set(r['home_team'] for r in rows) | set(r['away_team'] for r in rows))
score_chars = sorted(set(''.join(f"{r['home_score']}-{r['away_score']}" for r in rows)))

token_to_id = {}
id_to_token = {}

for name in teams + score_chars + ['|'] + RANK_TOKENS:
    i = len(token_to_id)
    token_to_id[name] = i
    id_to_token[i]    = name

BOS        = len(token_to_id)
vocab_size = BOS + 1
id_to_token[BOS] = '<BOS>'

print(f"teams: {len(teams)}  score chars: {len(score_chars)}  vocab size: {vocab_size}")

# ----------------------------
# Tokenize all matches
# Sequence: BOS, home, |, away, |, score_ch1, score_ch2, ..., BOS
# Score starts at position 4 (fixed — prompt is always 5 tokens long)
# ----------------------------

SCORE_START = 6

def tokenize(row):
    score   = f"{row['home_score']}-{row['away_score']}"
    h_rank  = token_to_id[rank_bucket(get_rank(row['home_team'], row['date']))]
    a_rank  = token_to_id[rank_bucket(get_rank(row['away_team'], row['date']))]
    toks    = [BOS, token_to_id[row['home_team']], h_rank, token_to_id['|'],
                    token_to_id[row['away_team']], a_rank, token_to_id['|']]
    toks   += [token_to_id[c] for c in score]
    toks   += [BOS]
    return toks

tokenized = [tokenize(r) for r in rows]
block_size = max(len(t) for t in tokenized)
print(f"max seq len / block_size: {block_size}")

sample_weights = []
for row in rows:
    year = int(row['date'][:4])
    t    = (year - min_year) / max(max_year - min_year, 1)
    sample_weights.append(math.exp(5 * t))

# ----------------------------
# Parameters
# ----------------------------

n_embd   = 32
n_head   = 4
n_layer  = 2
head_dim = n_embd // n_head

def matrix(nout, nin, std=0.08):
    return (torch.randn(nout, nin, device=device) * std).requires_grad_(True)

state_dict = {
    'wte':     matrix(vocab_size, n_embd),
    'wpe':     matrix(block_size, n_embd),
    'lm_head': matrix(vocab_size, n_embd),
}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)

num_params = sum(p.numel() for p in state_dict.values())
print(f"num params:  {num_params}")

# ----------------------------
# Model
# ----------------------------

def rmsnorm(x):
    return x * (x.pow(2).mean() + 1e-5).pow(-0.5)

def gpt(token_id, pos_id, keys, values):
    x = state_dict['wte'][token_id] + state_dict['wpe'][pos_id]
    x = rmsnorm(x)

    for li in range(n_layer):
        x_res = x
        x = rmsnorm(x)
        q = state_dict[f'layer{li}.attn_wq'] @ x
        k = state_dict[f'layer{li}.attn_wk'] @ x
        v = state_dict[f'layer{li}.attn_wv'] @ x
        keys[li].append(k)
        values[li].append(v)

        x_attn_parts = []
        for h in range(n_head):
            hs   = h * head_dim
            q_h  = q[hs:hs + head_dim]
            k_h  = torch.stack([ki[hs:hs + head_dim] for ki in keys[li]])
            v_h  = torch.stack([vi[hs:hs + head_dim] for vi in values[li]])
            attn_weights = F.softmax(k_h @ q_h / head_dim ** 0.5, dim=0)
            x_attn_parts.append(v_h.T @ attn_weights)

        x = state_dict[f'layer{li}.attn_wo'] @ torch.cat(x_attn_parts)
        x = x + x_res

        x_res = x
        x = rmsnorm(x)
        x = F.relu(state_dict[f'layer{li}.mlp_fc1'] @ x)
        x = state_dict[f'layer{li}.mlp_fc2'] @ x
        x = x + x_res

    return state_dict['lm_head'] @ x

# ----------------------------
# Optimizer
# ----------------------------

learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
optimizer = torch.optim.Adam(
    list(state_dict.values()), lr=learning_rate, betas=(beta1, beta2), eps=eps_adam
)

# ----------------------------
# Training loop
# ----------------------------

num_steps = 2000
print(f"\ntraining for {num_steps} steps...\n")

for step in range(num_steps):
    idx    = random.choices(range(len(tokenized)), weights=sample_weights, k=1)[0]
    tokens = tokenized[idx]
    n      = len(tokens) - 1

    keys   = [[] for _ in range(n_layer)]
    values = [[] for _ in range(n_layer)]
    losses = []

    for pos_id in range(n):
        logits = gpt(tokens[pos_id], pos_id, keys, values)
        if pos_id >= SCORE_START:
            losses.append(F.cross_entropy(
                logits.unsqueeze(0),
                torch.tensor([tokens[pos_id + 1]], device=device)
            ))

    if not losses:
        continue

    loss = torch.stack(losses).mean()
    optimizer.zero_grad()
    loss.backward()

    for pg in optimizer.param_groups:
        pg['lr'] = learning_rate * (1 - step / num_steps)

    optimizer.step()
    print(f"step {step+1:4d} / {num_steps} | loss {loss.item():.4f}", end='\r')

# ----------------------------
# Prediction
# ----------------------------

@torch.no_grad()
def predict_match(home, away, n_samples=50, temperature=0.7):
    if home not in token_to_id:
        print(f"  Unknown team: '{home}'")
        return {}
    if away not in token_to_id:
        print(f"  Unknown team: '{away}'")
        return {}

    latest = max(rank_snapshots)
    h_rank = token_to_id[rank_bucket(rank_snapshots[latest].get(home))]
    a_rank = token_to_id[rank_bucket(rank_snapshots[latest].get(away))]
    prompt = [BOS, token_to_id[home], h_rank, token_to_id['|'],
                   token_to_id[away], a_rank, token_to_id['|']]
    scores = {}

    for _ in range(n_samples):
        keys   = [[] for _ in range(n_layer)]
        values = [[] for _ in range(n_layer)]

        pos_id = 0
        for token_id in prompt:
            logits = gpt(token_id, pos_id, keys, values)
            pos_id += 1

        result = []
        token_id = torch.multinomial(F.softmax(logits / temperature, dim=0), 1).item()
        while pos_id < block_size and token_id != BOS:
            result.append(id_to_token[token_id])
            logits   = gpt(token_id, pos_id, keys, values)
            token_id = torch.multinomial(F.softmax(logits / temperature, dim=0), 1).item()
            pos_id  += 1

        score = ''.join(result)
        if score:
            scores[score] = scores.get(score, 0) + 1

    return scores

# ----------------------------
# Interactive match prediction
# ----------------------------

print(f"\n\n--- match predictor ---")
print(f"Known teams ({len(teams)} total): {', '.join(teams[:8])}, ...\n")

while True:
    home = input("Home team (or press Enter to quit): ").strip()
    if not home:
        break
    away = input("Away team: ").strip()
    if not away:
        break

    print(f"\nRunning 50 simulations for {home} vs {away}...\n")
    scores = predict_match(home, away, n_samples=50, temperature=0.7)

    if scores:
        print(f"  Predicted score distribution ({home} score first):")
        for score, count in sorted(scores.items(), key=lambda x: -x[1]):
            print(f"    {score:>10}  {'|' * count} ({count})")
        top = max(scores, key=scores.get)
        print(f"\n  Most likely result: {home} {top} {away}\n")
    else:
        print(f"  No valid scores generated — try more training steps.\n")

    if input("Predict another match? (y/n): ").strip().lower() != 'y':
        break

print("Done.")
