import random

# Parameters
NUM_JOBS = 10
NUM_MACHINES = 3
POP_SIZE = 6
GENS = 10
MUTATION_RATE = 0.2

processing_times = [random.randint(1, 20) for _ in range(NUM_JOBS)]

def fitness(chromosome):
    machine_loads = [0] * NUM_MACHINES
    for job, machine in enumerate(chromosome):
        machine_loads[machine] += processing_times[job]
    makespan = max(machine_loads)
    return 1 / makespan

def create_chromosome():
    return [random.randint(0, NUM_MACHINES - 1) for _ in range(NUM_JOBS)]

def initial_population():
    return [create_chromosome() for _ in range(POP_SIZE)]

def tournament_selection(population, fitness_values, k=3):
    selected = random.sample(list(zip(population, fitness_values)), k)
    return max(selected, key=lambda x: x[1])[0]

def crossover(parent1, parent2):
    point = random.randint(1, NUM_JOBS - 1)
    return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]

def mutate(chromosome):
    if random.random() < MUTATION_RATE:
        idx = random.randint(0, NUM_JOBS - 1)
        chromosome[idx] = random.randint(0, NUM_MACHINES - 1)
    return chromosome

population = initial_population()
print("Processing times:", processing_times)

for generation in range(GENS):
    fitness_values = [fitness(ch) for ch in population]
    new_population = []

    for _ in range(POP_SIZE // 2):
        parent1 = tournament_selection(population, fitness_values)
        parent2 = tournament_selection(population, fitness_values)
        child1, child2 = crossover(parent1, parent2)
        new_population.append(mutate(child1))
        new_population.append(mutate(child2))

    population = new_population

fitness_values = [fitness(ch) for ch in population]
best_index = fitness_values.index(max(fitness_values))
best_solution = population[best_index]

machine_loads = [0] * NUM_MACHINES
for job, machine in enumerate(best_solution):
    machine_loads[machine] += processing_times[job]

# print("\n--- Final Result ---")
print("Best Assignment:", best_solution)
print("Machine Loads:", machine_loads)
print("Makespan:", max(machine_loads))
