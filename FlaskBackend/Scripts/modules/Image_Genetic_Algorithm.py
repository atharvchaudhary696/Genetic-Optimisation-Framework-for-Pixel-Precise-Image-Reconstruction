import numpy as np
from PIL import Image
import itertools
import math
import random
from matplotlib import pyplot as plt
import os
import sys
from scipy import stats
# Defining Constants
# ------------------------------------------------------------------------------------------------------------------------------
# sol_per_population = 8 # Population size
# num_parents_mating = 4 # Mating pool size
# mutation_percent = .01 # Mutation percentage
# Data Representation
# ------------------------------------------------------------------------------------------------------------------------------

def chromosome2img(chromosome, img_shape):
    """
    Convert a 1D chromosome back to the original image shape.
    """
    return np.reshape(chromosome, img_shape)
# target_img = Image.open('IMAGES/fruit.jpg') # Open an image file
# target_arr = np.array(target_img) # Convert the image to a NumPy array
# img_shape = target_arr.shape
# # print(f"Image 3D Vector : {target_arr}")
# target_chromosome = img2chromosome(target_arr) # Convert 3D array to 1D array
# # print(f"Image 1D Vector : {target_chromosome}")
# restored_img_arr = chromosome2img(target_chromosome, img_shape)
# # print(f"Restored Image 3D Vector : {restored_img_arr}")
# Initialising the Population
# ------------------------------------------------------------------------------------------------------------------------------
def initial_population(img_shape, individual_count=8, dtype=np.uint8):
    # Generates a random initial population of individuals
    # Get the total number of pixels in an image.
    total_pixels = np.prod(img_shape)
    # Generate a random array of integers between 0 and 255.
    init_population = np.random.randint(0, 256, size=(
        individual_count, total_pixels), dtype=dtype)
    return init_population
def check_sufficient_parents(num_parents_mating, sol_per_population):
    # Calculate the number of ways to choose 2 parents from num_parents_mating parents.
    num_possible_permutations = len(list(itertools.permutations(
        iterable=np.arange(0, num_parents_mating), r=2)))
    # print(f'Iterable Method : {num_possible_permutations}')
    # num_possible_permutations = math.factorial(num_parents_mating) // (math.factorial(2) * math.factorial(num_parents_mating - 2))
    # print(f'Permutation Method : {num_possible_permutations}')
    num_required_permutations = sol_per_population - num_possible_permutations
    if num_required_permutations > num_possible_permutations:
        return False
    else:
        return True
# check = check_sufficient_parents(num_parents_mating, sol_per_population)
# if check == False:
#     print("\n*Inconsistency in the selected populatiton size or number of parents.*")
#     sys.exit(1)
# new_population = initial_population(img_shape, sol_per_population)
# print(new_population)
# Fitness Function
# ------------------------------------------------------------------------------------------------------------------------------
def fitness_fun(target_chrom, indiv_chrom):
    # The fitness is basicly calculated using the sum of absolute difference between genes values in the original and reproduced chromosomes
    fitness = np.mean(np.abs(target_chrom-indiv_chrom))
    # since we want higher fitness value, better the result instead of lower fitness value indicating better result as convention is to have maximising function
    fitness = np.sum(target_chrom) - fitness
    # print(f'Fitness value : {fitness}')
    return fitness
def calc_population_fitness(target_chrom, population):
    # Calculate the fitness of each solution in the population.
    quality = np.zeros(population.shape[0])
    for indv_num in range(population.shape[0]):
        # Calling fitness_fun(...) to get the fitness of the current solution.
        quality[indv_num] = fitness_fun(target_chrom, population[indv_num, :])
        # print(f'Quality : {quality}')
    return quality
# fit_quality = calc_population_fitness(target_chromosome, new_population)
def diversity_fitness_fun(target_chrom, indiv_chrom, population):
    # Calculate the fitness based on absolute difference
    fitness = np.mean(np.abs(target_chrom - indiv_chrom))

    # Calculate the average similarity to other individuals in the population
    similarity_penalty = np.mean(
        np.linalg.norm(population - indiv_chrom, axis=1))

    # Combine the fitness and similarity penalty
    fitness = fitness - 0.1 * similarity_penalty

    return fitness
def calc_population_diversity_fitness(target_chrom, population):
    quality = np.zeros(population.shape[0])
    for indv_num in range(population.shape[0]):
        quality[indv_num] = diversity_fitness_fun(
            target_chrom, population[indv_num, :], population)
    return quality
def entropy_fitness_fun(target_chrom, indiv_chrom):
    # Calculate the fitness based on absolute difference
    fitness = np.mean(np.abs(target_chrom - indiv_chrom))

    # Calculate the entropy of the individual chromosome
    indiv_entropy = stats.entropy(indiv_chrom.flatten(), base=2)

    # Combine the fitness and entropy terms
    # Adjust the weight of the entropy term
    fitness = fitness - 0.01 * indiv_entropy

    return fitness
def calc_population_entropy_fitness(target_chrom, population):
    quality = np.zeros(population.shape[0])
    for indv_num in range(population.shape[0]):
        quality[indv_num] = entropy_fitness_fun(
            target_chrom, population[indv_num, :])
    return quality
# Parent Selection for Mating
# ------------------------------------------------------------------------------------------------------------------------------
def selecting_mating_pool(num_parents, population, quality):
    # The array will have num_parents rows and population.shape[1] columns
    parents = np.empty((num_parents, population.shape[1]), dtype=np.uint8)
    """Retrieve the index of the best unselected solution in the population. 
    This is done by finding the index of the highest fitness value in the qualities array. 
    The numpy.where() function returns a tuple of arrays containing the indices of the elements in the qualities array that match the specified condition"""
    for parent_num in range(num_parents):
        # Retrieving the best unselected solution.
        max_indx = np.where(quality == np.max(quality))
        max_indx = max_indx[0][0]
        # Appending the currently selected
        parents[parent_num, :] = population[max_indx, :]
        """Set quality of selected individual to a negative value to not get 
        selected again. Algorithm calcululations will just make qualities >= 0.
        """
        quality[max_indx] = -1
    # print(parents)
    return parents
# parents = selecting_mating_pool(num_parents_mating, new_population, fit_quality)
# Single-point Crossover
# ------------------------------------------------------------------------------------------------------------------------------
def single_pt_crossover(parents, img_shape, individual_count):
    # Calculate the total number of elements in each image.
    num_elements = np.prod(img_shape)
    # Create an empty NumPy array to store the new population.
    new_population = np.empty(
        shape=(individual_count, num_elements), dtype=np.uint8)
    # print(f'Prod Method : {new_population}')
    # Previous parents (best elements)
    new_population[0:parents.shape[0], :] = parents
    # Measuring many offspring to be generated
    num_newly_generated = individual_count - parents.shape[0]
    # Getting all possible permutations of the selected parents.
    parents_permutations = list(itertools.permutations(
        iterable=np.arange(0, parents.shape[0]), r=2))

    # Randomly selecting the parents permutations to generate the offspring.
    selected_permutations = random.sample(
        range(len(parents_permutations)), num_newly_generated)
    # Randomly selecting the parents permutations to generate the offspring.
    # Create a list of all the indices of the permutations.
    permutation_indices = list(range(len(parents_permutations)))
    # Randomly shuffle the list of indices.
    random.shuffle(permutation_indices)
    # Select the first `num_newly_generated` indices.
    selected_permutations = permutation_indices[:num_newly_generated]
    # print(f'Shuffle Method : {selected_permutations}')
    # selected_permutations = random.sample(range(len(parents_permutations)), num_newly_generated)
    # print(f'Sample Method : {selected_permutations}')

    # Initialize the index for the offspring in the new population
    comb_indx = parents.shape[0]

    # Loop through each selected permutation of parents
    for comb in range(len(selected_permutations)):
        # Get the index of the selected permutation
        selected_comb_indx = selected_permutations[comb]
        # Retrieve the actual combination of parents based on the selected index
        selected_comb = parents_permutations[selected_comb_indx]
        # Calculate the halfway point of the genes in each chromosome
        half_size = np.int32(new_population.shape[1] / 2)
        # Apply crossover by exchanging half of the genes between two parents
        new_population[comb_indx + comb,
                       0:half_size] = parents[selected_comb[0], 0:half_size]
        new_population[comb_indx + comb,
                       half_size:] = parents[selected_comb[1], half_size:]

    # Return the updated population
    return new_population
def multi_pt_crossover(parents, img_shape, individual_count):
    num_elements = np.prod(img_shape)
    new_population = np.empty(
        shape=(individual_count, num_elements), dtype=np.uint8)
    new_population[0:parents.shape[0], :] = parents

    num_newly_generated = individual_count - parents.shape[0]

    for i in range(num_newly_generated):
        # Randomly select multiple parents for crossover
        selected_parents = random.sample(range(parents.shape[0]), 3)

        # Sort the selected parents to get crossover points
        selected_parents.sort()
        crossover_points = [random.randint(0, num_elements) for _ in range(2)]
        crossover_points.sort()

        # Perform multi-point crossover
        new_population[parents.shape[0] + i, :crossover_points[0]
                       ] = parents[selected_parents[0], :crossover_points[0]]
        new_population[parents.shape[0] + i, crossover_points[0]:crossover_points[1]
                       ] = parents[selected_parents[1], crossover_points[0]:crossover_points[1]]
        new_population[parents.shape[0] + i, crossover_points[1]:] = parents[selected_parents[2], crossover_points[1]:]

    return new_population
# new_population = single_pt_crossover(parents, img_shape, sol_per_population)
# print(new_population)
# Mutation
# ------------------------------------------------------------------------------------------------------------------------------
def mutation(population, num_parents_mating, mut_percent):
    for indx in range(num_parents_mating, population.shape[0]):
        # A predefined percent of genes are selected randomly.
        rand_indx = np.uint32(np.random.random(size=np.uint32(
            mut_percent/100*population.shape[1])) * population.shape[1])
        # Changing the values of the selected genes randomly.
        new_values = np.uint8(np.random.random(size=rand_indx.shape[0]) * 256)
        # Updating population after mutation.
        population[indx, rand_indx] = new_values
    # print(population)
    return population
def enhanced_mutation(population, num_parents_mating, mut_percent):
    for indx in range(num_parents_mating, population.shape[0]):
        # A predefined percent of genes are selected randomly.
        rand_indx = np.uint32(np.random.random(size=np.uint32(
            mut_percent/100*population.shape[1])) * population.shape[1])

        # Introduce color mutation
        # Assume images have 3 color channels (R, G, B)
        color_indices = rand_indx % 3
        population[indx, rand_indx] += np.uint8(
            np.random.randint(-20, 20, size=rand_indx.shape[0]))
        population[indx, rand_indx] = np.clip(
            population[indx, rand_indx], 0, 255)

        # Non-uniform mutation
        mutation_strength = np.random.random(
            size=rand_indx.shape[0]) * 2.0 - 1.0  # Random values between -1 and 1
        population[indx, rand_indx] += np.uint8(
            mutation_strength * population[indx, rand_indx])
        population[indx, rand_indx] = np.clip(
            population[indx, rand_indx], 0, 255)

    return population
def brush_stroke_mutation(population, num_parents_mating, mut_percent, brush_stroke_size):
    for indx in range(num_parents_mating, population.shape[0]):
        # A predefined percent of genes are selected randomly.
        rand_indx = np.uint32(np.random.random(size=np.uint32(
            mut_percent/100*population.shape[1])) * population.shape[1])

        # Introduce color mutation
        # Assume images have 3 color channels (R, G, B)
        color_indices = rand_indx % 3
        population[indx, rand_indx] += np.uint8(
            np.random.randint(-20, 20, size=rand_indx.shape[0]))
        population[indx, rand_indx] = np.clip(
            population[indx, rand_indx], 0, 255)

        # Non-uniform mutation
        mutation_strength = np.random.random(
            size=rand_indx.shape[0]) * 2.0 - 1.0  # Random values between -1 and 1
        population[indx, rand_indx] += np.uint8(
            mutation_strength * population[indx, rand_indx])
        population[indx, rand_indx] = np.clip(
            population[indx, rand_indx], 0, 255)

        # Add brush strokes
        brush_start = np.random.randint(
            0, population.shape[1] - brush_stroke_size)
        brush_end = brush_start + brush_stroke_size
        population[indx, brush_start:brush_end] = np.uint8(
            np.random.randint(0, 256, size=brush_stroke_size))

    return population
# new_population = mutation(new_population, num_parents_mating, mutation_percent)
# Saving Images
# ------------------------------------------------------------------------------------------------------------------------------
def save_images(curr_iteration, qualities, new_population, im_shape,
                save_point, save_dir):
    """
    Saving best solution in a given generation as an image in the specified directory.
    Images are saved accoirding to stop points to avoid saving images from 
    all generations as saving mang images will make the algorithm slow.
    """
    if (np.mod(curr_iteration, save_point) == 0):
        # Selecting best solution (chromosome) in the generation.
        best_solution_chrom = new_population[np.where(qualities ==
                                                      np.max(qualities))[0][0], :]
        # Decoding the selected chromosome to return it back as an image.
        best_solution_img = chromosome2img(best_solution_chrom, im_shape)
        # Saving the image in the specified directory.
        plt.imsave(save_dir+'1.jpg', best_solution_img)
        plt.imsave(save_dir+'2.jpg', best_solution_img)
        plt.imsave(save_dir+'3.jpg', best_solution_img)
# ------------------------------------------------------------------------------------------------------------------------------
# Main Loop
# ------------------------------------------------------------------------------------------------------------------------------
def Main(parameters_list,folderPath):
    sol_per_population = 8  # Population size
    num_parents_mating = 4  # Mating pool size
    mutation_percent = .01  # Mutation percentage

    target_img, target_arr, img_shape, target_chromosome = parameters_list
    # print(f"Image 3D Vector : {target_arr}")

    
    # print(f"Image 1D Vector : {target_chromosome}")

    restored_img_arr = chromosome2img(target_chromosome, img_shape)
    # print(f"Restored Image 3D Vector : {restored_img_arr}")

    check = check_sufficient_parents(num_parents_mating, sol_per_population)
    if check == False:
        print("\n*Inconsistency in the selected populatiton size or number of parents.*")
        sys.exit(1)
    new_population = initial_population(img_shape, sol_per_population)

    for generation in range(100000):
        # Calculating the fitness of each individual in the population
        fit_quality = calc_population_fitness(target_chromosome, new_population)
        # fit_quality = calc_population_diversity_fitness(target_chromosome, new_population)
        # fit_quality = calc_population_entropy_fitness(target_chromosome, new_population)
        print('Best Fitness value : ', np.max(
            fit_quality), ', Generation : ', generation)

        # Selecting the best parents in the population for mating
        parents = selecting_mating_pool(
            num_parents_mating, new_population, fit_quality)

        # Generating next generation using crossover
        # new_population = single_pt_crossover(parents, img_shape, sol_per_population)
        new_population = multi_pt_crossover(parents, img_shape, sol_per_population)

        # Applying mutation for offspring
        # new_population = mutation(new_population, num_parents_mating, mutation_percent)
        new_population = enhanced_mutation(
            new_population, num_parents_mating, mutation_percent)
        # new_population = brush_stroke_mutation(new_population, num_parents_mating, mutation_percent, brush_stroke_size=100)

        # Save best individual in the generation as an image for later visualization.
        save_images(generation, fit_quality, new_population, img_shape,
                    save_point=5000, save_dir=folderPath+'//')
def Main1(parameters_list,folderpath):
    print("working")
if __name__ == "__main__":
    import ImageParameters
    image_path="img\\2.jpg"
    parameters_list = ImageParameters.Main(image_path)
    Main(parameters_list)
