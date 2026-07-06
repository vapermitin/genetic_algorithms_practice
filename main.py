# Для заданного неориентированного графа G = (V,E)
# необходимо найти множество вершин S,
# которое является вершинным покрытием графа и минимально

from math import inf
import random

k = 100
adjacency_matrix = [[1 if i != j else 0 for j in range(k)] for i in range(k)]


class GeneticAlgorithm:
    def __init__(self, input_matrix, population_size, hard_constraint_penalty,
                 crossover_probability, mutation_probability, tournament_size,
                 elitement_size):
        self.input_matrix = input_matrix
        self.population_size = population_size
        self.hard_constraint_penalty = hard_constraint_penalty
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.tournament_size = tournament_size
        self.elitement_size = elitement_size
    
    def generate_random_population(self, count):
        result = []
        for _ in range(count):
            random_genotype = bin(random.randrange(1, 2 ** len(self.input_matrix)))[2:].zfill(len(self.input_matrix))
            result.append(list(map(int, random_genotype)))
        return result
    
    def fitness_function(self, individual):
        penalty_edges = 0
        for i in range(len(self.input_matrix)):
            for j in range(i, len(self.input_matrix)):
                if self.input_matrix[i][j] != 1:
                    continue
                if individual[i] == 0 and individual[j] == 0:
                    penalty_edges += 1
        return len(self.input_matrix) - sum(individual) - penalty_edges * self.hard_constraint_penalty
    
    # Равномерное скрещивание
    def crossover_function(self, first_individual, second_individual):
        for index in range(len(self.input_matrix)):
            first_individual_value = first_individual[index]
            if random.random() < 0.5:
                first_individual[index] = second_individual[index]
            if random.random() < 0.5:
                second_individual[index] = first_individual_value
    
    def mutation_function(self, individual):
        for index in range(len(self.input_matrix)):
            if random.random() < self.mutation_probability:
                individual[index] = 1 - individual[index]

    def run(self, iterations):
        current_population = self.generate_random_population(self.population_size)
        max_fitness = -inf
        max_fitness_individual = None

        for iteration in range(iterations):
            fitnesses = [self.fitness_function(ind) for ind in current_population]

            sorted_indices = sorted(range(len(current_population)),
                                    key=lambda i: fitnesses[i], reverse=True)
            elite_indices = sorted_indices[:self.elitement_size]
            new_population = [current_population[i].copy() for i in elite_indices]

            while len(new_population) < self.population_size:
                idx1 = random.sample(range(len(current_population)), self.tournament_size)
                parent1_idx = max(idx1, key=lambda i: fitnesses[i])
                idx2 = random.sample(range(len(current_population)), self.tournament_size)
                parent2_idx = max(idx2, key=lambda i: fitnesses[i])

                child1 = current_population[parent1_idx].copy()
                child2 = current_population[parent2_idx].copy()

                if random.random() < self.crossover_probability:
                    self.crossover_function(child1, child2)
                self.mutation_function(child1)
                self.mutation_function(child2)

                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)

            current_population = new_population

            new_fitnesses = [self.fitness_function(ind) for ind in current_population]
            population_best = max(range(len(new_fitnesses)), key=lambda index: new_fitnesses[index])
            if new_fitnesses[population_best] > max_fitness:
                max_fitness = new_fitnesses[population_best]
                max_fitness_individual = current_population[population_best]
            print(f"Итерация: {iteration}, макс. приспособленность: {max_fitness}")

        return max_fitness_individual


og = GeneticAlgorithm(
    adjacency_matrix,
    population_size=100,
    hard_constraint_penalty=10,
    crossover_probability=0.8,
    mutation_probability=0.01,
    tournament_size=10,
    elitement_size=2
)
print(og.run(100))
