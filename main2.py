import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random

class InteractionDataset(Dataset):
    def __init__(self, user_item_pairs, num_items):
        self.data = user_item_pairs
        self.num_items = num_items

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user, pos_item = self.data[idx]
        return user, pos_item

def load_movielens_100k():
    df = pd.read_csv("https://files.grouplens.org/datasets/movielens/ml-100k/u.data", sep='\t', header=None)
    df.columns = ["user", "item", "rating", "timestamp"]
    df = df[df['rating'] >= 4]
    df['user'] -= 1
    df['item'] -= 1
    num_users = df['user'].max() + 1
    num_items = df['item'].max() + 1  # <-- FIXED
    interactions = list(zip(df['user'], df['item']))
    return interactions, num_users, num_items


class Generator(nn.Module):
    def __init__(self, num_users, num_items, emb_dim):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, emb_dim)
        self.item_embedding = nn.Embedding(num_items, emb_dim)
        self.item_bias = nn.Embedding(num_items, 1)

    def forward(self, user_ids):
        pu = self.user_embedding(user_ids)
        qi = self.item_embedding.weight
        bi = self.item_bias.weight.squeeze()
        scores = torch.matmul(pu, qi.T) + bi
        attention = F.softmax(scores, dim=1)
        return attention, scores

class Discriminator(nn.Module):
    def __init__(self, num_users, num_items, emb_dim):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, emb_dim)
        self.item_embedding = nn.Embedding(num_items, emb_dim)
        self.item_bias = nn.Embedding(num_items, 1)

    def forward(self, user_ids, attention):
        vu = self.user_embedding(user_ids)
        wi = self.item_embedding.weight
        ci = self.item_bias.weight.squeeze()
        virtual_item = torch.matmul(attention, wi)
        virtual_bias = torch.matmul(attention, ci)
        return torch.sum(vu * virtual_item, dim=1) + virtual_bias

def bpr_loss(pos_scores, neg_scores):
    return -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8))

interactions, num_users, num_items = load_movielens_100k()
dataset = InteractionDataset(interactions, num_items)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

embedding_dim = 32
batch_size = 64
epochs = 10
lr = 0.01
tau = 0.5
L = 100
l = 10

G = Generator(num_users, num_items, embedding_dim)
D = Discriminator(num_users, num_items, embedding_dim)
opt_G = torch.optim.Adam(G.parameters(), lr=lr)
opt_D = torch.optim.Adam(D.parameters(), lr=lr)

def sample_attentive(attention, top_L, l, tau):
    batch_size = attention.size(0)
    result = torch.zeros_like(attention)
    for i in range(batch_size):
        topk = torch.topk(attention[i], top_L).indices
        probs = F.softmax(attention[i][topk] / tau, dim=0)
        indices = topk[torch.multinomial(probs, l, replacement=True)]
        result[i][indices] = probs[torch.arange(l)]
    return result

for epoch in range(epochs):
    G.train(); D.train(); total_loss = 0
    for user_ids, pos_items in dataloader:
        attn, _ = G(user_ids)
        sampled_attn = sample_attentive(attn, L, l, tau)

        vu = D.user_embedding(user_ids)
        vi = D.item_embedding(pos_items)
        bi = D.item_bias(pos_items).squeeze()
        pos_scores = torch.sum(vu * vi, dim=1) + bi
        neg_scores = D(user_ids, sampled_attn.detach())

        opt_D.zero_grad()
        loss_D = bpr_loss(pos_scores, neg_scores)
        loss_D.backward(); opt_D.step()

        attn, _ = G(user_ids)
        sampled_attn = sample_attentive(attn, L, l, tau)
        neg_scores = D(user_ids, sampled_attn)
        loss_G = -bpr_loss(pos_scores.detach(), neg_scores)

        opt_G.zero_grad()
        loss_G.backward(); opt_G.step()
        total_loss += loss_D.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

print("Training complete.")

def recommend_top_k(generator, user_id, known_items, k=10):
    generator.eval()
    user_tensor = torch.tensor([user_id])
    with torch.no_grad():
        attention, _ = generator(user_tensor)  # (1, num_items)
        scores = attention.squeeze()           # (num_items,)

        # Filter out known items
        scores[known_items] = -1e9             # mask known items

        top_k = torch.topk(scores, k).indices.tolist()
        return top_k

test_user = 2
user_interacted_items = [item for user, item in interactions if user == test_user]
top_recommendations = recommend_top_k(G, test_user, user_interacted_items, k=10)

print(f"Top 10 recommendations for user {test_user}: {top_recommendations}")
