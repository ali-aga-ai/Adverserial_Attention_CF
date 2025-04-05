import torch
import torch.nn as nn

# Matrix Factorization model
class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, k):
        super().__init__()
        # Each user and item gets a k-dimensional latent vector
        self.user_factors = nn.Embedding(num_users, k) 
        self.item_factors = nn.Embedding(num_items, k)
        self.bias_factors = nn.Embedding(num_items, 1) # Bias factors for items

    def forward(self, user_ids, item_ids): #forward is an oveerridden function of nn module which basicallytakes input to nn and returns output, thus if u do model = MatrixFactorization() and model(user_ids, item_ids) it will return the output of the nn module (basically pytorch abstracts away the forward fn and calls it when you call the model)
        u = self.user_factors(user_ids)  # Get user latent vectors
        v = self.item_factors(item_ids)  # Get item latent vectors
        b = self.bias_factors(item_ids)  # Get item bias factors
        return (u * v).sum(1) + b.squeeze(1) #bias is added to the dot product of user and item latent vectors, sum(1) sums over the k dimension, and squeeze(1) removes the extra dimension from the bias factor

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Sample training data (fake)
user_ids = torch.tensor([0, 1, 0], dtype=torch.long).to(device)     # Users
item_ids = torch.tensor([1, 2, 2], dtype=torch.long).to(device)     # Items
ratings = torch.tensor([5.0, 3.0, 4.0], dtype=torch.float).to(device)  # Ratings

def train(model, user_ids, item_ids, ratings):
    
    criterion = nn.MSELoss()                            # Mean Squared Error loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Adam optimizer, optimizers are algorithms that update the weights of the model based on the gradients computed during backpropagation, Adam is a popular choice for training deep learning models. for simple gradient descent you could use SGD

    # Training loop
    for epoch in range(1000):
        model.train()                           # Set model to training mode (not necessary, but good practice)
        preds = model(user_ids, item_ids)      # Predict ratings
        loss = criterion(preds, ratings)       # Compute loss vs true ratings

        optimizer.zero_grad()                  # Clear existing gradients from previous steps.
        loss.backward()                        # Computes gradients of the loss w.r.t. all trainable parameters
        optimizer.step()                       # updates the parameters using the gradients from .backward()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")  # Print loss every 100 epochs

