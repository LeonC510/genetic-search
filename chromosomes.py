import numpy as np


class Chromosome:
    def __init__(self, search_space, fitness_function, rng=np.random.default_rng(), location=None,
                 mutate_distribution_scale=2):
        # Initialize random number generator.
        self.rng = rng

        # Copy location provided to the class attribute. If no location is specified, then generate a random location.
        if location is not None:
            self.location = list(location)
        else:
            # Referenced Numpy.Random.Generator.Uniform — NumPy v2.1 Manual (n.d.).
            self.location = [self.rng.uniform(search_space[0][0], search_space[0][1]),
                             self.rng.uniform(search_space[1][0], search_space[1][1])]

        # Check whether search_space is properly defined.
        for dimension, _ in enumerate(search_space):
            if search_space[dimension][0] >= search_space[dimension][1]:
                # Referenced Built-in Exceptions (n.d.) when choosing the error to raise.
                raise ValueError("Search space size on dimension " + str(dimension) + " is equal or smaller than 0")
        self.search_space = search_space

        self.mutate_distribution_scale = mutate_distribution_scale

        self.fitness_function = fitness_function

    def fitness(self):
        return self.fitness_function(self.location[0], self.location[1])

    def crossover(self, other_chromosome):
        child_location = list()
        for dimension, value in enumerate(self.location):
            child_location.append((value + other_chromosome.location[dimension]) / 2)

        return Chromosome(self.search_space, self.fitness_function, self.rng, child_location, self.mutate_distribution_scale)

    # Referenced Marnach (2012).
    def __add__(self, other):
        return self.crossover(other)

    def mutate(self):
        # Iterate through the dimensions of the chromosome's location.
        for dimension, _ in enumerate(self.location):
            # We mutate the chromosome by adjusting its location. The amount of mutation is determined randomly,
            # with a higher probability for a small mutation, as determined by the exponential probability
            # distribution. We multiply by the size of the search space to scale the random mutation amount. We also
            # multiply it by a Rademacher random variable so that the location can both increase and decrease. Used
            # Numpy.Random.Exponential — NumPy v2.0 Manual (n.d.) to sample from an exponential random distribution.
            # Also referenced Random Generator — NumPy v2.1 Manual (n.d.) to use the NumPy random number generator.
            self.location[dimension] += ((self.rng.exponential(self.mutate_distribution_scale)
                                          * (self.search_space[dimension][1]
                                             - self.search_space[dimension][0]))
                                         * (1 if self.rng.random() < 0.5 else -1))

            # Trim in case of over-adjustment.
            if self.location[dimension] <= self.search_space[dimension][0]:
                self.location[dimension] = self.search_space[dimension][0]
            elif self.location[dimension] >= self.search_space[dimension][1]:
                self.location[dimension] = self.search_space[dimension][1]
