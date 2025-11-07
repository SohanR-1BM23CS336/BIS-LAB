import numpy as np
from skimage import data, color, filters, img_as_float
from concurrent.futures import ThreadPoolExecutor

img = color.rgb2gray(img_as_float(data.coins()))
rows, cols = img.shape
population_size = 10
population = [np.random.rand(rows, cols) for _ in range(population_size)]

def fitness(solution):
    edges = filters.sobel(solution)
    return np.sum(edges * img)

def mutate(solution, rate=0.01):
    mask = np.random.rand(*solution.shape) < rate
    noise = np.random.rand(*solution.shape)
    return np.clip(solution + mask * (noise - 0.5), 0, 1)

def crossover(a, b):
    mask = np.random.rand(*a.shape) > 0.5
    return np.where(mask, a, b)

def evolve_cell(i):
    cell = population[i]
    neighbors = [population[(i - 1) % population_size], population[(i + 1) % population_size]]
    parent = max(neighbors, key=fitness)
    child = mutate(crossover(cell, parent))
    return child if fitness(child) > fitness(cell) else cell

for _ in range(20):
    with ThreadPoolExecutor() as executor:
        population = list(executor.map(evolve_cell, range(population_size)))

best = max(population, key=fitness)
import matplotlib.pyplot as plt
plt.imshow(filters.sobel(best), cmap='gray')
plt.axis('off')
plt.show()
