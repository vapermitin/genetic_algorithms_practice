from genetic_algorithm import *
from matplotlib import pyplot as plt

k = 100
ITERATIONS_COUNT = 100
adjacency_matrix = [[1 if i != j else 0 for j in range(k)] for i in range(k)]

ga = GeneticAlgorithm(
    adjacency_matrix,
    population_size=100,
    hard_constraint_penalty=10,
    crossover_probability=0.8,
    mutation_probability=0.01,
    tournament_size=10,
    elitement_size=2
)

_, fitnesses = ga.run(ITERATIONS_COUNT)
plt.figure(figsize=(6, 4))
plt.plot(list(range(1, ITERATIONS_COUNT + 1)), fitnesses)
plt.show()
