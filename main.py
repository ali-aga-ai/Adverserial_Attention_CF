import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
from matrixFactorization import train, MatrixFactorization
from generator import Generator, Discriminator, GAN

m = 3 # Number of users
n = 4 # Number of items
interactions_matrix = torch.randint(0, 2, (m, n))

#converting the m by n matrix into user_ids, item_ids and ratings
user_ids = []
item_ids = [] 
ratings = []
for i in range(m):
    for j in range(n):
        user_ids.append(i)
        item_ids.append(j)
        if interactions_matrix[i][j] == 1:  
            ratings.append(1.0)  # Assuming a rating of 1 for interaction
        else:
            ratings.append(0.0)

# Convert to PyTorch tensors
user_ids = torch.tensor(user_ids, dtype=torch.long)
item_ids = torch.tensor(item_ids, dtype=torch.long)
ratings = torch.tensor(ratings, dtype=torch.float)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MatrixFactorization(num_users=3, num_items=4, k=5).to(device)  # Instantiate model

train(model, user_ids, item_ids, ratings)  # Train the model

test_user_tensor = torch.tensor([0], dtype=torch.long).to(device)  # Test user ID
user_embedding = model.user_factors(test_user_tensor) # converts the user tensor into an embedding of k = 5 dimensions
print(user_embedding)  # Print user embedding

temp = []
for i in range(n): # Get item embeddings for all items
    test_item_tensor = torch.tensor([i], dtype=torch.long).to(device)  # Test item ID
    item_embedding = model.item_factors(test_item_tensor)  # converts the item tensor into an embedding of k = 5 dimensions
    temp.append(item_embedding)
item_embeddings_matrix = torch.cat(temp, dim=0)  # Concatenate item embeddings into a matrix

print(item_embeddings_matrix)  # Print item embeddings matrix
print(item_embeddings_matrix.shape)  # Print shape of item embeddings matrix

for i in range(m):
    for j in range(n):
        user_tensor = torch.tensor([i], dtype=torch.long).to(device)
        item_tensor = torch.tensor([j], dtype=torch.long).to(device)
        user_embedding = model.user_factors(user_tensor)
        item_embedding = model.item_factors(item_tensor)
        item_bias = model.bias_factors(item_tensor)
        print(f"Complete data: User {i}, Item {j}, User Embedding: {user_embedding}, Item Embedding: {item_embedding}, Item Bias: {item_bias}")
        print(f"Shape data: User {i}, Item {j}, User Embedding: {user_embedding.shape}, Item Embedding: {item_embedding.shape}, Item Bias: {item_bias.shape}")

        generator = Generator(user_embedding, item_embeddings_matrix.T, item_bias)
        generator_output = generator.forward()

        a = F.softmax(generator_output, dim = 0)

        discriminator = Discriminator(a, item_embeddings_matrix, item_bias) # items bias could give issues
        virtual_item, virtual_item_bias = discriminator.forward()
        score_for_virtual_item = torch.matmul(virtual_item, user_embedding.T)  # removed bias for virtual item

        score_for_actual_item = torch.matmul(item_embedding, user_embedding.T) + item_bias

        print(f"Generator Output: {generator_output}, Discriminator Output: {virtual_item}, Score for Virtual Item: {score_for_virtual_item}, Score for Actual Item: {score_for_actual_item}")
        print(f"Generator Output Shape: {generator_output.shape}, Discriminator Output Shape: {virtual_item.shape}, Score for Virtual Item Shape: {score_for_virtual_item.shape}, Score for Actual Item Shape: {score_for_actual_item.shape}, virtual_item_bias Shape: {virtual_item_bias.shape}")
print("End of loop")