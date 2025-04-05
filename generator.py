import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    '''
    the input tensor is a 1 * d tensor where d is the number of latent dimensions and it represents the user
    the input matrix is d * n matrix where n is the number of items / movies and d is the number of latent dimensions
    output is a 1 * n matrix representing the attention score for each item / movie for the user
    '''
    def __init__(self, input_tensor, input_matrix, input_bias):
        super().__init__()
        self.input_tensor = input_tensor
        self.input_matrix = input_matrix
        self.input_bias = input_bias
    
    def forward(self):
        # Perform matrix multiplication between input_tensor and input_matrix
        output = torch.matmul(self.input_tensor, self.input_matrix)
        
        # Add the input_bias to the output
        output += self.input_bias
        
        return output
    
class Discriminator(nn.Module):
    '''
    input tensor would be 1 * n where n is the number of items / movies
    input matrix is n * d where n is the number of items / movies and d is the number of latent dimensions
    OUTPUT IS THE VIRTUAL ITEM WHICH IS 1 * d where d is the number of latent dimensions
    discriminator bias is 1 * n where n is the number of items / movies
    '''
    def __init__(self, input_tensor, input_matrix, discriminator_bias):
        super().__init__()
        self.input_tensor = input_tensor
        self.input_matrix = input_matrix
        self.discriminator_bias = discriminator_bias
    
    def forward(self):
        # Perform matrix multiplication between input_tensor and input_matrix
        output = torch.matmul(self.input_tensor, self.input_matrix)

        # temp = []
        # for i,j in zip(self.input_tensor[0], self.discriminator_bias[0]):
        #     temp.append(i * j)
        # discriminator_bias_value = torch.tensor(temp, dtype=torch.float32).reshape(1, -1)  # Reshape to 1 * n
        discriminator_bias_value = (self.input_tensor[0] * self.discriminator_bias[0]).unsqueeze(0) #ALTERNATIVE WAY TO DO THE SAME THING

        
        return output, discriminator_bias_value
    
class GAN(nn.Module):  
    def __init__(self, generator, discriminator, user_tensor):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.user_tensor = user_tensor
    
    def forward(self):
        # Generate virtual items using the generator
        virtual_items = self.generator.forward()
        
        # Discriminate between real and virtual items
        virtual_item, virtual_item_bias = self.discriminator.forward()
        # virtual_item is a 1 * d tensor where d is the number of latent dimensions
        score_for_virtual_item = torch.matmul(virtual_item, self.user_tensor) + virtual_item_bias
        
        return virtual_item, score_for_virtual_item
    

user_embedding = torch.tensor([1.0,2.0,3.0], dtype=torch.float32)  # Shape: (1,3) (1 user with 3 latent dimensions)
items_embedding_matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32)  # Shape: (3,2) (2 movies with 3 latent dimensions) 
items_bias_generator = torch.tensor([1.0, 2.0], dtype=torch.float32)  # Shape: (1,2) (bias for 2 movies))
items_bias_discriminator = torch.tensor([1.0, 2.0], dtype=torch.float32)  # Shape: (1,2) (bias for 2 movies))

generator = Generator(user_embedding, items_embedding_matrix, items_bias_generator)
output = generator.forward() 
print(output)
a = F.softmax(output, dim = 0)
print(a)  # Output shape: (1,2) (1 user with 2 movies)

discriminator = Discriminator(a, items_embedding_matrix.T, items_bias_discriminator)
virtual_item, discriminator_bias_vector = discriminator.forward()
print(virtual_item)  # Output shape: (1,3) (1 user with 3 latent dimensions)

gan = GAN(generator, discriminator, user_embedding)
v_tem, score = gan.forward()
print(v_tem, score)  # Output shape: (1,3) (1 user with 3 latent dimensions)



