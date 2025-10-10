import random
import math

def fitness_function(x):
  return x**2

population_size = 100
num_genes = 8
mutation_rate = 0.01
crossover_rate = 0.8
num_generations = 5

population = [[random.randint(0, 1) for _ in range(num_genes)] for _ in range(population_size)]

def binary_to_decimal(binary_chromosome):
    decimal_value = 0
    for bit in binary_chromosome:
        decimal_value = decimal_value * 2 + bit
    return decimal_value

def evaluate_fitness(population):
    fitness_values = []
    for chromosome in population:
        x = binary_to_decimal(chromosome)
        fitness = fitness_function(x)
        fitness_values.append(fitness)
    return fitness_values

def select_parents(population, fitness_values):
    total_fitness = sum(fitness_values)
    if total_fitness == 0:
        return random.sample(population, 2)

    selection_probabilities = [f / total_fitness for f in fitness_values]
    parents_indices = random.choices(range(len(population)), weights=selection_probabilities, k=2)
    return [population[parents_indices[0]], population[parents_indices[1]]]

def crossover(parent1, parent2, crossover_rate):
    if random.random() < crossover_rate:
        point = random.randint(1, num_genes - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    else:
        return parent1, parent2

def mutate(chromosome, mutation_rate):
    mutated_chromosome = chromosome[:]
    for i in range(num_genes):
        if random.random() < mutation_rate:
            mutated_chromosome[i] = 1 - mutated_chromosome[i]
    return mutated_chromosome

best_solution = None
best_fitness = -math.inf;

'''print("Starting Gene Expression Algorithm...")'''

for generation in range(num_generations):
    fitness_values = evaluate_fitness(population)

    current_best_fitness = max(fitness_values)
    current_best_index = fitness_values.index(current_best_fitness)
    current_best_individual_binary = population[current_best_index]
    current_best_individual_decimal = binary_to_decimal(current_best_individual_binary)

    if current_best_fitness > best_fitness:
        best_fitness = current_best_fitness
        best_solution = current_best_individual_binary
        best_individual_result = current_best_individual_decimal

    print(f"Generation {generation + 1}: Best Fitness = {current_best_fitness}, Best Individual (decimal representation) = {current_best_individual_decimal}")

    new_population = []
    for _ in range(population_size // 2):
        parent1, parent2 = select_parents(population, fitness_values)
        child1, child2 = crossover(parent1, parent2, crossover_rate)
        new_population.append(mutate(child1, mutation_rate))
        new_population.append(mutate(child2, mutation_rate))

    population = new_population
