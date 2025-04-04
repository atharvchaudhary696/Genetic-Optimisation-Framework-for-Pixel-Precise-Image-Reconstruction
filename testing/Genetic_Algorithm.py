# Given : Y = w1x1 + w2x2 + w3x3 + w4x4 + w5x5 + w6x6 and inputs values are (x1,x2,x3,x4,x5,x6)=(4,-2,7,5,11,1)
# Using Genetic Approach to find maximise the output

# Steps involved 
# --------------------------------------------------------------------------------------------------------------
"""
    1. The fitness of each solution in the population is calculated.
    2. The best individuals (parents) are selected based on their fitness.
    3. The next generation is generated through crossover.
    4. Some variations are introduced to the offspring through mutation.
    5. The new population is created based on the selected parents and mutated offspring.
"""

# Input Paramters
# ---------------------------------------------------------------------------------------------------------------
equation_inputs = [4, -2, 3.5, 5, -11, -4.7]  # The input values for the linear equation
num_weights = 6  # The number of weights to be optimized

# Initialise Population
# ----------------------------------------------------------------------------------------------------------------
import numpy

sol_per_pop = 8  # Number of solutions (individuals) in the population
num_parents_mating = 4
# Defining the population size.
pop_size = (
    sol_per_pop,
    num_weights,
)  # Size of the population, a 2D array with sol_per_pop rows and num_weights columns

new_population = numpy.random.uniform(
    low=-4.0, high=4.0, size=pop_size
)  # Randomly initialized population within the specified range

# print(new_population)


# Fitness Function
# ------------------------------------------------------------------------------------------------------------------
def cal_pop_fitness(equation_inputs, pop):
    """Calculating the fitness value of each solution in the current population.
       The fitness function calculates the sum of products between each input and its corresponding weight."""
    fitness = numpy.sum(pop * equation_inputs, axis=1)  # sum along the columns (axis=1)
    return fitness


# Selection of the Fittest
# -------------------------------------------------------------------------------------------------------------------
def select_mating_pool(pop, fitness, num_parents):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    parents = numpy.empty((num_parents, pop.shape[1])) # initializes an empty matrix to store the selected parents
    for parent_num in range(num_parents):
        # The index of the solution with the maximum fitness is found using numpy.where(), and the corresponding row (solution) is added to the parents matrix
        max_fitness_idx = numpy.where(fitness == numpy.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999999  # The fitness of the selected parent is set to a very low value to avoid selecting it again
    return parents


# Crossover for Mating
# ----------------------------------------------------------------------------------------------------------------------
def crossover(parents, offspring_size):
    offspring = numpy.empty(offspring_size)
    # The point at which crossover takes place between two parents. Usually, it is at the center.
    crossover_point = numpy.uint8(offspring_size[1] / 2) # The crossover point is set at the midpoint of the genes (weights)

    for k in range(offspring_size[0]):
        # Index of the first parent to mate.
        parent1_idx = k % parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k + 1) % parents.shape[0]
        # The new offspring will have its first half of its genes taken from the first parent.
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # The new offspring will have its second half of its genes taken from the second parent.
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring


# Mutation
# -------------------------------------------------------------------------------------------------------------------------
"""For each offspring, a random value is generated from a uniform distribution between -1.0 and 1.0. 
   This random value is added to the gene at index 4 (fifth gene) of each offspring. This introduces a small random variation in the offspring's genes"""
def mutation(offspring_crossover):
    # Mutation changes a single gene in each offspring randomly.
    for idx in range(offspring_crossover.shape[0]):
        # The random value to be added to the gene.
        random_value = numpy.random.uniform(-1.0, 1.0, 1)
        offspring_crossover[idx, 4] = offspring_crossover[idx, 4] + random_value
    return offspring_crossover


# Main Loop
# --------------------------------------------------------------------------------------------------------------------------
num_generations = 10000

for generation in range(num_generations):
    print("Generation : ", generation)
    # Measuring the fitness of each chromosome in the population.
    fitness = cal_pop_fitness(equation_inputs, new_population)
    # print(fitness)
    # Selecting the best parents in the population for mating.
    parents = select_mating_pool(new_population, fitness, num_parents_mating)
    # Generating next generation using crossover.
    offspring_crossover = crossover(
        parents, offspring_size=(pop_size[0] - parents.shape[0], num_weights)
    )
    # Adding some variations to the offsrping using mutation.
    offspring_mutation = mutation(offspring_crossover)
    # Creating the new population based on the parents and offspring.
    new_population[0 : parents.shape[0], :] = parents
    new_population[parents.shape[0] :, :] = offspring_mutation
    # The best result in the current iteration.
    print(
        "Best result : ", numpy.max(numpy.sum(new_population * equation_inputs, axis=1))
    )

    # Getting the best solution after iterating finishing all generations.
    # At first, the fitness is calculated for each solution in the final generation.
    fitness = cal_pop_fitness(equation_inputs, new_population)

    # Then return the index of that solution corresponding to the best fitness.
    best_match_idx = numpy.where(fitness == numpy.max(fitness))

print("Best solution : ", new_population[best_match_idx, :])
print("Best solution fitness : ", fitness[best_match_idx])
