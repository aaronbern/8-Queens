# Aaron Bernard
# Programming Assignment 2
# CS441

import tkinter as tk
import random
import time
import matplotlib.pyplot as plt

# Genetic Algorithm Functions
def create_individual():
    return [random.randint(0, 7) for _ in range(8)]

def fitness(individual):
    n = len(individual)
    maxFitness = 28  # 8 queens can form 28 non-attacking pairs
    
    horizontal_collisions = sum([individual.count(queen) - 1 for queen in individual]) // 2

    main_diagonal_collisions = 0
    anti_diagonal_collisions = 0

    main_diagonals = [0] * (2 * n - 1)
    anti_diagonals = [0] * (2 * n - 1)

    for i in range(n):
        main_diagonals[i - individual[i] + (n - 1)] += 1
        anti_diagonals[i + individual[i]] += 1

    for count in main_diagonals:
        if count > 1:
            main_diagonal_collisions += (count - 1)

    for count in anti_diagonals:
        if count > 1:
            anti_diagonal_collisions += (count - 1)

    total_collisions = horizontal_collisions + main_diagonal_collisions + anti_diagonal_collisions
    return int(maxFitness - total_collisions)

def probability(individual, fitness):
    return fitness(individual) / 28  # Max fitness is 28

def random_pick(population, probabilities):
    total = sum(probabilities)
    r = random.uniform(0, total)
    upto = 0
    for i, probability in enumerate(probabilities):
        if upto + probability >= r:
            return population[i]
        upto += probability
    assert False, "Shouldn't get here"

def reproduce(x, y):
    n = len(x)
    c = random.randint(0, n - 1)
    return x[:c] + y[c:]

def mutate(individual):
    n = len(individual)
    c = random.randint(0, n - 1)
    m = random.randint(0, n - 1)
    individual[c] = m
    return individual

# Visualization Functions
def draw_board(canvas, individual):
    canvas.delete("all")
    for row in range(8):
        for col in range(8):
            color = "white" if (row + col) % 2 == 0 else "black"
            canvas.create_rectangle(col*50, row*50, (col+1)*50, (row+1)*50, fill=color)
    for col, row in enumerate(individual):
        canvas.create_oval(col*50 + 10, row*50 + 10, col*50 + 40, row*50 + 40, fill="red")

def genetic_algorithm_visualized(canvas, population, fitness, mutation_probability=0.3, generations=1000, elite_size=2, random_restart_interval=30):
    maxFitness = 28
    fitness_history = []
    initial_population = population[0]
    first_solution = None

    for generation in range(generations):
        population.sort(key=fitness, reverse=True)
        new_population = population[:elite_size]  # Carry over the best individuals

        probabilities = [probability(n, fitness) for n in population]
        for _ in range((len(population) - elite_size) // 2):
            parent1 = random_pick(population, probabilities)
            parent2 = random_pick(population, probabilities)
            child1 = reproduce(parent1, parent2)
            child2 = reproduce(parent2, parent1)
            if random.random() < mutation_probability:
                child1 = mutate(child1)
            if random.random() < mutation_probability:
                child2 = mutate(child2)
            new_population.append(child1)
            new_population.append(child2)

        population = new_population

        best_individual = max(population, key=fitness)
        current_fitness = fitness(best_individual)
        fitness_history.append(current_fitness)
        draw_board(canvas, best_individual)
        canvas.update()
        time.sleep(0.1)  # Slow down the visualization for observation

        print(f"Generation {generation}: Best Fitness = {current_fitness}")
        if current_fitness == maxFitness and first_solution is None:
            first_solution = best_individual.copy()
            print(f"First perfect solution found in generation {generation}: {first_solution}")

        if current_fitness == maxFitness:
            print(f"Perfect solution found in generation {generation}: {best_individual}")
            break  # This ensures the loop stops once the perfect solution is found

        if generation % random_restart_interval == 0:
            # Introduce new random individuals to the population
            new_individuals = [create_individual() for _ in range(len(population) // 2)]
            population[-len(new_individuals):] = new_individuals

    return population, fitness_history, initial_population, first_solution

# Main Function
def main():
    root = tk.Tk()
    root.title("8-Queens Genetic Algorithm Visualization")
    canvas = tk.Canvas(root, width=400, height=400)
    canvas.pack()

    population_size = 300  # Further increased population size for more diversity
    generations = 1000
    mutation_probability = 0.3  # Further increased mutation probability for more exploration
    random_restart_interval = 30  # More frequent random restarts

    population = [create_individual() for _ in range(population_size)]
    solution, fitness_history, initial_population, first_solution = genetic_algorithm_visualized(canvas, population, fitness, mutation_probability, generations, elite_size=2, random_restart_interval=random_restart_interval)

    best_individual = max(solution, key=fitness)
    print("Initial population state:", initial_population)
    print("First solution found:", first_solution)
    print("Best solution found:", best_individual)
    print("Fitness of best solution:", fitness(best_individual))

    plt.plot(fitness_history)
    plt.title("Fitness over Generations")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.show()

    root.mainloop()

if __name__ == "__main__":
    main()
