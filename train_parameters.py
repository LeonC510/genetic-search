import sys

import numpy as np
from search import genetic_search
from error import error

# How many times to attempt to find the optimal parameter combination.
param_search_iterations = 10
# Maximum number of steps to take to find the local optimum parameter combination.
param_search_max_steps = 300

# Population size to use for the genetic algorithm.
population_size = 100
# Maximum number of generations that the genetic algorithm will perform for.
max_generations = 200
# Termination condition for the genetic algorithm.
termination_static_max_fitness_iterations = 5

# Number of trials to perform to gauge accuracy.
trials = 100
# Count results returned by the algorithm that has an associated error less than this value as a successful run.
success_threshold = -0.2

# Search range and step sizes for each parameter.
survival_rate_search_range = [0.0, 0.5]
mutate_distribution_scale_search_range = [0.0000000000001, 0.01]
param_search_ranges = [survival_rate_search_range, mutate_distribution_scale_search_range]
survival_rate_step_size = 0.05
mutate_distribution_scale_step_size = 0.001
param_search_step_sizes = [survival_rate_step_size, mutate_distribution_scale_step_size]

def accuracy_rate(survival_rate, mutate_distribution_scale):
    """Measure the accuracy rate of the genetic algorithm with the parameters passed.

    :param survival_rate: The survival rate parameter for the genetic algorithm.
    :param mutate_distribution_scale: The mutate distribution scale parameter for the genetic algorithm.
    :return: The accuracy rate of the genetic algorithm, measured by simulation.
    """
    successes = 0
    for _ in range(trials):
        location = genetic_search(population_size, max_generations, survival_rate, mutate_distribution_scale,
                                  termination_static_max_fitness_iterations)
        if error(location[0], location[1]) < success_threshold:
            successes += 1

    return successes / trials


# The following code is largely adapted from the answer to question 4.
def round_to_step_size(num, step_size):
    """Round the provided `num` to the nearest multiple of `step_size`."""
    # The following line is partly adapted from Turra (2011).
    return round(round(num / step_size) * step_size, 4)


def move(parameters, step_sizes):
    """Take a step in a direction that increases the accuracy rate.

    :param parameters: List of the parameters for the genetic algorithm.
    :param step_sizes: List of step sizes for each parameter, respectively.
    :return: Adjusted parameters.
    """
    current_accuracy_rate = accuracy_rate(parameters[0], parameters[1])

    # We try to move the specified step size in every direction of every dimension. If one move yields a better
    # accuracy rate, we make that move by returning the parameter combination with that move applied.
    if (accuracy_rate(parameters[0] + step_sizes[0], parameters[1])
            > current_accuracy_rate):
        parameters[0] = parameters[0] + step_sizes[0]
    elif (accuracy_rate(parameters[0] - step_sizes[0], parameters[1])
          > current_accuracy_rate):
        parameters[0] = parameters[0] - step_sizes[0]
    elif (accuracy_rate(parameters[0], parameters[1] + step_sizes[1])
          > current_accuracy_rate):
        parameters[1] = parameters[1] + step_sizes[1]
    elif (accuracy_rate(parameters[0], parameters[1] - step_sizes[1])
          > current_accuracy_rate):
        parameters[1] = parameters[1] - step_sizes[1]

    return parameters


def optimize_parameters(iterations, max_steps):
    """Search for the optimum combination of parameters using multiple hill-climbing.

    This docstring is written referencing Goodger & Rossum (2001).

    Positional arguments:
    iterations -- how many times the algorithm performs simple hill-climbing.
    step_size -- the step size for the simple hill-climbing portion of this algorithm.
    max_steps -- the maximum number of steps that the simple hill-climbing portion of this algorithm will take.
    """
    print("Training started.")

    # Create dictionaries to store the local and global optimums.
    local_maximums = dict()
    global_maximums = set()

    # Initialize random number generator. Referenced Random Generator — NumPy v2.1 Manual (n.d.).
    rng = np.random.default_rng()

    # We run the simple hill-climbing algorithm multiple times with random starting points.
    for iteration in range(iterations):
        # Generate random starting position in the range specified. We round the random numbers to the nearest
        # multiple of step size to keep the algorithm from operating on different grids for each starting position.
        # I referenced Random — Generate Pseudo-Random Numbers (n.d.) to write the code generating the random numbers.
        start_positions = [
            round_to_step_size(
                rng.uniform(param_search_ranges[0][0], param_search_ranges[0][1]), param_search_step_sizes[0]),
            round_to_step_size(
                rng.uniform(param_search_ranges[1][0], param_search_ranges[1][1]), param_search_step_sizes[1])]

        for step in range(max_steps):
            print(f"Training... {round((iteration / iterations + step / max_steps / iterations) * 100, 2)}% complete.")
            new_position = move(start_positions, param_search_step_sizes)
            # Round to step size to avoid floating-point numbers' precision loss problem.
            for dimension, value in enumerate(new_position):
                new_position[dimension] = round_to_step_size(value, param_search_step_sizes[dimension])

            # Check whether the new position is the same as the old one. If so, the local minimum has been found.
            same_position = True
            for dimension, new_value in enumerate(new_position):
                if new_value != start_positions[dimension]:
                    same_position = False
            if same_position:
                # Terminate this nested simple hill-climbing algorithm.
                break

            start_positions = new_position

        # Store the local minimum found in this iteration.
        local_maximums.update({tuple(start_positions): accuracy_rate(start_positions[0], start_positions[1])})

    print("Training complete!")

    current_largest_local_maximum = 0
    # Iterate through all the local minimums to find the global minimum.
    # I learned from fiveobjects (2021) how to iterate through a dictionary.
    for location, maximum_accuracy in local_maximums.items():
        # If the minimum in the current iteration is smaller than all previously detected, we update the list.
        if maximum_accuracy > current_largest_local_maximum:
            # Count the current minimum as the current smallest minimum.
            current_largest_local_maximum = maximum_accuracy
            # Clear all the contents of global_maximums, as they are all proven to not be the global minimum.
            # This line is adapted from Python Set Clear() Method (n.d.)
            global_maximums.clear()
            # Save the current minimum as a global minimum. Referenced Python - Add Set Items (n.d.).
            global_maximums.add(tuple([location, maximum_accuracy]))
        # If there is a second global optimum, we also add that to our set of global minimums.
        elif maximum_accuracy == current_largest_local_maximum:
            global_maximums.add(tuple([location, maximum_accuracy]))

    return global_maximums


best_parameters = optimize_parameters(param_search_iterations, param_search_max_steps)
for index, combo in enumerate(best_parameters):
    print(f"Combo {index} with {round(combo[1] * 100, 4)}% accuracy: Survival rate = {combo[0][0]}; Mutation distribution scale = {combo[0][1]}")
