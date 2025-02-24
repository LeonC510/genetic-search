import numpy as np
from pandas import DataFrame, concat
from chromosomes import Chromosome
from error import error

def genetic_search(population_size = 100,
                   max_generations = 100,
                   survival_rate = 0.3,
                   mutate_distribution_scale = 0.001,
                   termination_static_max_fitness_iterations = 10):
    search_space = [[-3, 3], [-3, 3]]
    current_running_max_fitness = 0
    static_max_fitness_iterations = 0

    # Initialize random number generator. Referenced Random Generator — NumPy v2.1 Manual (n.d.).
    rng = np.random.default_rng()

    population = DataFrame(columns=["chromosome", "fitness"])
    for _ in range(population_size):
        # Create a new chromosome. The fitness function is defined as 2^-error because fitness needs to be maximized,
        # while error needs to be minimized.
        chromosome = Chromosome(search_space, lambda a, b: 2**(-error(a, b)), rng, mutate_distribution_scale=mutate_distribution_scale)
        # Referenced Pandas.DataFrame — Pandas 1.4.4 Documentation (n.d.).
        new_row = DataFrame(columns=["chromosome", "fitness"], data=[[chromosome, chromosome.fitness()]])
        # Add the new row to the population. Referenced Pandas.Concat — Pandas 1.4.4 Documentation (n.d.).
        population = concat([population, new_row])
    # Sort the population by fitness. Referenced Pandas.DataFrame.Sort_values — Pandas 2.2.3 Documentation (n.d.).
    population.sort_values("fitness", ascending=False, inplace=True)

    # Initialize the running max fitness for generation 0. Inspired by Pandas.Series — Pandas 2.2.3 Documentation (n.d.)
    # to use .iloc here.
    current_running_max_fitness = population["fitness"].iloc[0]

    for generation in range(max_generations):
        # Select parents.
        # Calculate the probability each chromosome should be selected based on fitness. How Do I Select a Subset of a
        # DataFrame? — Pandas 2.2.3 Documentation (n.d.) helped me select the `fitness` column.
        selection_probabilities = population["fitness"] / population["fitness"].sum()
        # Sample chromosomes from the population to be parents. Referenced Numpy.Random.Choice — NumPy v2.1 Manual (n.d.).
        parents = rng.choice(population["chromosome"],
                             size=round(population_size * (1 - survival_rate)) * 2,
                             replace=True,
                             p=selection_probabilities)

        # Create children.
        children = list()
        # Slice the parents list into two. Chromosomes from each side would be paired to create children.
        parent_1s = parents[:int(len(parents) / 2)]
        parent_2s = parents[int(len(parents) / 2):]
        for index, parent_1 in enumerate(parent_1s):
            children.append(parent_1 + parent_2s[index])

        # Replace the least fit older generation with the children. We slice the previous population to select the most fit
        # individuals, and concatenate the children to the population list. Referenced Pandas.DataFrame.Iloc — Pandas 2.2.3
        # Documentation (n.d.) to slice the population.
        population = population.iloc[:len(population) - len(children)]
        for child in children:
            new_row = DataFrame(columns=["chromosome", "fitness"], data=[[child, child.fitness()]])
            population = concat([population, new_row])

        # Mutate each individual. The size of the mutation is determined randomly.
        for chromosome in population["chromosome"]:
            chromosome.mutate()

        # Sort the population by fitness. Referenced Pandas.DataFrame.Sort_values — Pandas 2.2.3 Documentation (n.d.).
        population.sort_values("fitness", ascending=False, inplace=True)

        # Count the number of iterations where the maximum fitness was static.
        if population["fitness"].iloc[0] == current_running_max_fitness:
            static_max_fitness_iterations += 1
        # If the maximum fitness changed from the last iteration, we reset the counter and store the current maximum
        # fitness.
        else:
            current_running_max_fitness = population["fitness"].iloc[0]
            static_max_fitness_iterations = 0

        if static_max_fitness_iterations >= termination_static_max_fitness_iterations:
            break

    # Return the best location found by the algorithm.
    return population["chromosome"].iloc[0].location
