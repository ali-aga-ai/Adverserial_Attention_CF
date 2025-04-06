import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from tqdm import tqdm
import random

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBEDDING_DIM = 64
TOP_L = 100   # Candidate pool size
TOP_l = 10    # Attentive items sampled per user
TEMP = 0.2    # Temperature for softmax sampling
EPOCHS = 20
BATCH_SIZE = 1024
TOP_K = 10    # For evaluation

def load_movielens_100k():
    df = pd.read_csv("https://files.grouplens.org/datasets/movielens/ml-100k/u.data", sep='\t', header=None)
    df.columns = ["user", "item", "rating", "timestamp"]
    df = df[df['rating'] >= 4]  # implicit feedback

    user_map = {u: i for i, u in enumerate(df['user'].unique())}
    item_map = {i: j for j, i in enumerate(df['item'].unique())}
    df['user'] = df['user'].map(user_map)
    df['item'] = df['item'].map(item_map)

    num_users = df['user'].nunique()
    num_items = df['item'].nunique()

    train_data = defaultdict(set)
    test_data = defaultdict(set)

    for user, group in df.groupby('user'):
        items = list(group['item'])
        random.shuffle(items)
        cutoff = int(0.8 * len(items))
        train_items = items[:cutoff]
        test_items = items[cutoff:]
        train_data[user] = set(train_items)
        test_data[user] = set(test_items)

    return train_data, test_data, num_users, num_items

train_data, test_data, num_users, num_items = load_movielens_100k()

class Generator(nn.Module):
    def __init__(self, num_users, num_items, embed_dim):
        super().__init__()
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.item_embed = nn.Embedding(num_items, embed_dim)
        self.item_bias = nn.Embedding(num_items, 1)

    def forward(self, user_ids):
        p_u = self.user_embed(user_ids)  # (B, d)
        q_i = self.item_embed.weight     # (n_items, d)
        b_i = self.item_bias.weight.squeeze()  # (n_items)
        scores = torch.matmul(p_u, q_i.t()) + b_i  # (B, n_items)
        return scores  # r_ui

class Discriminator(nn.Module):
    def __init__(self, num_users, num_items, embed_dim):
        super().__init__()
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.item_embed = nn.Embedding(num_items, embed_dim)
        self.item_bias = nn.Embedding(num_items, 1)

    def predict(self, user_ids, item_ids):
        v_u = self.user_embed(user_ids)
        w_i = self.item_embed(item_ids)
        c_i = self.item_bias(item_ids).squeeze()
        return (v_u * w_i).sum(dim=1) + c_i

    def predict_virtual(self, user_ids, attn_weights, item_ids_subset):
        v_u = self.user_embed(user_ids)
        W = self.item_embed(item_ids_subset)  # (B, l, d)
        C = self.item_bias(item_ids_subset).squeeze(-1)  # (B, l)

        attn = F.softmax(attn_weights / TEMP, dim=1)  # (B, l)

        w_virtual = torch.sum(attn.unsqueeze(-1) * W, dim=1)  # (B, d)
        c_virtual = torch.sum(attn * C, dim=1)  # (B)

        return (v_u * w_virtual).sum(dim=1) + c_virtual

def sample_attentive_items(gen_scores, L=TOP_L, l=TOP_l):
    topk_vals, topk_idx = torch.topk(gen_scores, L, dim=1)
    probs = F.softmax(topk_vals / TEMP, dim=1)
    sampled_idx = torch.multinomial(probs, l, replacement=False)
    final_items = torch.gather(topk_idx, 1, sampled_idx)
    return final_items  # (B, l)

def get_train_pairs(train_data):
    pairs = []
    for u, items in train_data.items():
        for i in items:
            pairs.append((u, i))
    return pairs

class TrainDataset(Dataset):
    def __init__(self, user_item_pairs):
        self.pairs = user_item_pairs
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        return self.pairs[idx]

def evaluate(generator, test_data, train_data, top_k=TOP_K):
    generator.eval()
    HR, NDCG = [], []
    for user in test_data:
        test_items = list(test_data[user])
        if not test_items: continue

        scores = generator(torch.tensor([user], device=DEVICE)).detach().cpu().numpy().flatten()
        seen = train_data[user]
        scores[list(seen)] = -np.inf

        top_items = np.argpartition(scores, -top_k)[-top_k:]
        top_items = top_items[np.argsort(-scores[top_items])]

        hits = [1 if item in test_items else 0 for item in top_items]
        HR.append(np.sum(hits))
        NDCG.append(np.sum([hit / np.log2(idx+2) for idx, hit in enumerate(hits)]))
    return np.mean(HR), np.mean(NDCG)

gen = Generator(num_users, num_items, EMBEDDING_DIM).to(DEVICE)
disc = Discriminator(num_users, num_items, EMBEDDING_DIM).to(DEVICE)
opt_gen = torch.optim.Adam(gen.parameters(), lr=0.001)
opt_disc = torch.optim.Adam(disc.parameters(), lr=0.001)

train_pairs = get_train_pairs(train_data)
train_loader = DataLoader(TrainDataset(train_pairs), batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(EPOCHS):
    gen.train(); disc.train()
    total_loss_d, total_loss_g = 0.0, 0.0

    for users, pos_items in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        users = users.to(DEVICE)
        pos_items = pos_items.to(DEVICE)

        # Generator forward pass
        gen_scores = gen(users)  # (B, n_items)
        attn_items = sample_attentive_items(gen_scores)  # (B, l)
        attn_weights = torch.gather(gen_scores, 1, attn_items)

        disc.zero_grad()
        y_pos = disc.predict(users, pos_items)
        y_virtual = disc.predict_virtual(users, attn_weights, attn_items)
        loss_d = -torch.mean(torch.log(torch.sigmoid(y_pos - y_virtual)))
        loss_d.backward()
        opt_disc.step()
        total_loss_d += loss_d.item()

        gen.zero_grad()
        gen_scores = gen(users)
        attn_items = sample_attentive_items(gen_scores)
        attn_weights = torch.gather(gen_scores, 1, attn_items)

        y_pos = disc.predict(users, pos_items).detach()
        y_virtual = disc.predict_virtual(users, attn_weights, attn_items)
        loss_g = -torch.mean(torch.log(torch.sigmoid(y_pos - y_virtual)))
        loss_g.backward()
        opt_gen.step()
        total_loss_g += loss_g.item()

    hr, ndcg = evaluate(gen, test_data, train_data)
    print(f"Epoch {epoch+1}: Disc Loss={total_loss_d:.4f}, Gen Loss={total_loss_g:.4f}, HR@{TOP_K}={hr:.4f}, NDCG@{TOP_K}={ndcg:.4f}")
