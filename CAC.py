import random
import numpy as np
import math
import time
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

# Decoding
def decode(p, q, h_b, w_b):
    p_value = []
    q_value = []
    for i in range(len(p)):
        p_value.append(0)
        for j in range(h_b):
            p_value[i] += p[i][j] * 2**(h_b - 1 - j)
    for i in range(len(p)):
        q_value.append(0)
        for j in range(w_b):
            q_value[i] += q[i][j] * 2**(w_b - 1 - j)
    return p_value, q_value

# Create Chromosomes
def conc(p, q, N):
    chromosome_value = []
    for i in range(N):
        chromosome_value.append(p[i] + q[i])
    return chromosome_value

def encode_block(block):
    flat_block = block.flatten()
    if np.all(flat_block == 0):
        return '01'  # All pixels are black
    elif np.all(flat_block == 1):
        return '1'   # All pixels are white
    else:
        encoded_values = ''.join(map(str, flat_block))
        return '00 ' + encoded_values

def constant_area_coding(image_array, kernel_height, kernel_width):
    height, width = image_array.shape
    encoded_image = []

    for i in range(0, height, kernel_height):
        row_encoding = []
        for j in range(0, width, kernel_width):
            # Get the current block
            block = image_array[i:i + kernel_height, j:j + kernel_width]
            encoded_block = encode_block(block)
            row_encoding.append(encoded_block)
        encoded_image.append(row_encoding)
    return encoded_image

# Fitness function
def fitness(p_value, q_value, image):
    fitness_value = [-1 for _ in range(len(p_value))]
    best_convolved_image = None
    for i in range(len(p_value)):
        kernel_h, kernel_w = p_value[i], q_value[i]

        # Note we ignore the cases where p%h != 0 and q%w != 0 for plotting reasons but you cann uncomment the line bellow to include them
        #if kernel_h != 0 and kernel_w != 0 and kernel_h <= image.shape[0]) :
        if kernel_h != 0 and kernel_w != 0 and kernel_h <= image.shape[0] and (image.shape[1]%kernel_w   ==0 ) and (image.shape[0]%kernel_h  == 0 ) :
            convolved_image = constant_area_coding(image, kernel_h, kernel_w)
            n1 = image.size
            n2 = sum(len(row) for row in convolved_image)
            fitness_value[i] = n1 / n2
            if best_convolved_image is None or fitness_value[i] > max(fitness_value):
                best_convolved_image = convolved_image
        else:
            print(f'Value at {i} is omitted: kernel_h={kernel_h}, kernel_w={kernel_w}')
    return fitness_value, best_convolved_image

# Select parents
def sorting(fitness_value, chromosome_value,p,q, N):
    selected_populations = []
    sorted_index = np.argsort(fitness_value)
    k = int(N / 2)
    selected_populations = np.array(chromosome_value)[sorted_index[-k:]]
    p = np.array(p)[sorted_index[-k:]].tolist()
    q = np.array(q)[sorted_index[-k:]].tolist()
    return selected_populations.tolist(),p,q

# Crossover
def crossover(parent1, parent2, offsprings):
    length = len(parent1)
    crossover_point = np.random.randint(1, length-1) 
    # Create offspring by swapping portions after the crossover point
    offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
    offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
    offsprings.append(offspring1)
    offsprings.append(offspring2)
    return offsprings

# Mutation
def mutate(individual):
    mask = np.random.randint(len(individual))
    individual[mask] = 1 - individual[mask]  # Flip the bit
    return individual

# Decode the image
def decode_image(encoded_image, kernel_height, kernel_width, h, w):
    decoded_image = np.ones((h, w), dtype=np.uint8)
    for i in range(0, h, kernel_height):
        for j in range(0, w, kernel_width):
            encoded_block = encoded_image[i // kernel_height][j // kernel_width]
            if encoded_block == '01':
                decoded_image[i:i + kernel_height, j:j + kernel_width] = 0
            elif encoded_block == '1':
                decoded_image[i:i + kernel_height, j:j + kernel_width] = 1
            else:
                flat_block = list(map(int, encoded_block.split(' ')[1]))
                decoded_block = np.array(flat_block).reshape((kernel_height, kernel_width))
                decoded_image[i:i + kernel_height, j:j + kernel_width] = decoded_block
    return decoded_image




# Main Genetic Algorithm
def main():
    h = 10
    w = 10
    h_b = math.ceil(math.log2(h))
    w_b = math.ceil(math.log2(w))
    N = 8
    epochs = 10
    random.seed(time.time())

    # Generate a random black-and-white image
    image = np.random.randint(0, 2, (h, w))
    plt.imshow(image, cmap='gray')
    plt.title('Random Black and White Image')
    plt.show()

    p = [[random.randint(0, 1) for _ in range(h_b)] for _ in range(N)]
    q = [[random.randint(0, 1) for _ in range(w_b)] for _ in range(N)]

    for epoch in range(epochs):
        p_value, q_value = decode(p, q, h_b, w_b)

        print('Original P, Q:')
        print(f'P: {p_value}')
        print(f'Q: {q_value}')

        chromosome_value = conc(p, q, N)
        fitness_value, best_convolved_image = fitness(p_value, q_value, image)
        print(f'Fitness at epoch {epoch}: {fitness_value}')

        chromosome_value,p,q = sorting(fitness_value, chromosome_value,p,q, N)

        if epoch != epochs - 1:
            offsprings = []
            for i in range(0, int(N / 2), 2):
                crossover(chromosome_value[i], chromosome_value[i + 1], offsprings)

            for i in range(int(N / 2)):
                offsprings[i] = mutate(offsprings[i])

            for i in range(int(N / 2)):
                chromosome_value.append(offsprings[i])

            p = []
            q = []
            for chromosome in chromosome_value:
                p.append(chromosome[:h_b])
                q.append(chromosome[h_b:])

        else:
            offsprings = []
            for i in range(0, int(N / 2), 2):
                crossover(chromosome_value[i], chromosome_value[i + 1], offsprings)

            for i in range(int(N / 2)):
                offsprings[i] = mutate(offsprings[i])

            for i in range(int(N / 2)):
                chromosome_value.append(offsprings[i])

            p = []
            q = []
            for chromosome in chromosome_value:
                p.append(chromosome[:h_b])
                q.append(chromosome[h_b:])

            fitness_value, best_convolved_image = fitness(p_value, q_value, image)
            fitness_value.sort(reverse=True)
            print(f'Fitness at last epoch: {fitness_value}')
            chromosome_value,p,q, = sorting(fitness_value, chromosome_value,p,q, N/2)

            p_value, q_value = decode(p, q, h_b, w_b)
            img = decode_image(best_convolved_image, p_value[0], q_value[0], h, w)
            plt.imshow(img, cmap='gray')
            plt.title('Output Image')
            plt.show()

        print('New P, Q:')
        print(f'P: {p_value}')
        print(f'Q: {q_value}')
        print(f'End of epoch: {epoch}')
        print('######################################################################')

main()
