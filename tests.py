from genetic_algorithm import *

for k in range(1, 50):
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

    best_individual, _ = ga.run(100)
    assert sum(best_individual) == k - 1
