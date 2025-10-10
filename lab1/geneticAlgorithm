import random

def convert_to_binary_manual(number):
  if number == 0:
    return "0"

  binary_string = ""
  while number > 0:
    remainder = number % 2
    binary_string = str(remainder) + binary_string
    number //= 2
  return binary_string

def convert_to_decimal_manual(binary_string):
    decimal_value = 0
    for digit in binary_string:
        decimal_value = decimal_value * 2 + int(digit)
    return decimal_value

def fitness(x):
  return x**2

initial = []
x = []
f = []
probability = []
expected_count = []
n = 4

for i in range(n):
  x.append(int(input(f"Enter the x value for individual {i+1}: ")))

num_generations = int(input("Enter the number of generations: "))

for val in x:
  initial.append(convert_to_binary_manual(val))

for generation in range(num_generations):
    print(f"\n--- Generation {generation + 1} ---")

    f = [fitness(val) for val in x]

    total_fitness = sum(f)
    probability = [fit / total_fitness for fit in f]
    expected_count = [prob * n for prob in probability]

    mating_pool = random.choices(initial, weights=probability, k=n)
    print("Mating Pool (binary):", mating_pool)

    offspring_binary = []
    for i in range(0, n, 2):
        parent1 = mating_pool[i]
        parent2 = mating_pool[i+1]
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        offspring_binary.extend([child1, child2])

    print("Offspring (binary):", offspring_binary)

    # Convert offspring binary to decimal and update x for the next generation
    x = [convert_to_decimal_manual(binary_str) for binary_str in offspring_binary]
    initial = offspring_binary # Update initial with the new binary strings
    print("Offspring (decimal):", x)

# After all generations, print the final best individual and its fitness
final_fitness = [fitness(val) for val in x]
best_individual_index = final_fitness.index(max(final_fitness))
print(f"\n--- Final Result ---")
print(f"Best individual (decimal): {x[best_individual_index]}")
print(f"Best individual (binary): {initial[best_individual_index]}")
print(f"Best fitness: {final_fitness[best_individual_index]}")
