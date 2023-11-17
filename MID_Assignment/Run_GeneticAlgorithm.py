import os
import numpy as np
import pandas as pd
# Benchmark fonksiyonları ve sınırları
def ackley(x):
    return -20 * np.exp(-0.2 * np.sqrt(np.mean(x**2))) - np.exp(np.mean(np.cos(2 * np.pi * x))) + 20 + np.exp(1)

def griewank(x):
    return np.sum(x**2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1)))) + 1

def schwefel(x):
    return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))

def rastrigin(x):
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

def sphere(x):
    return np.sum(x**2)

def perm(x):
    d = len(x)
    return np.sum((np.sum((j + 10) * (x[j] - 1) for j in range(d)))**2)

def zakharov(x):
    return np.sum(x**2) + (np.sum(0.5 * np.arange(1, len(x) + 1) * x))**2 + (np.sum(0.5 * np.arange(1, len(x) + 1) * x))**4

def rosenbrock(x):
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def dixon_price(x):
    return (x[0] - 2)**2 + np.sum((2 * x[1:]**2 - x[:-1])**2)

# Genetik Algoritma
def genetic_algorithm(objective_function, pop_size, num_generations, mutation_rate, crossover_rate, dim):
    # Popülasyonu rastgele başlat
    population = np.random.uniform(-32.768, 32.768, size=(pop_size, dim))

    for generation in range(num_generations):
        # Fitness değerlerini hesapla
        fitness_values = np.array([objective_function(ind) for ind in population])

        # Seçim ve çaprazlama
        selected_indices = tournament_selection(fitness_values, 2)
        parent1 = population[selected_indices[0]]
        parent2 = population[selected_indices[1]]
        child = crossover(parent1, parent2, crossover_rate)

        # Mutasyon
        child = mutate(child, mutation_rate)

        # Yeni çocuğu popülasyona ekle
        min_fitness_index = np.argmin(fitness_values)
        population[min_fitness_index] = child

    # En iyi çözümü ve fitness değerini döndür
    best_index = np.argmin(fitness_values)
    best_solution = population[best_index]

    return best_solution, fitness_values[best_index]

def tournament_selection(fitness_values, k):
    # Turnuva seçimi uygula
    selected_indices = np.random.choice(len(fitness_values), k, replace=False)
    return selected_indices

def crossover(parent1, parent2, rate):
    # Tek nokta çaprazlama
    if np.random.rand() < rate:
        crossover_point = np.random.randint(1, len(parent1))
        child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    else:
        child = parent1
    return child

def mutate(child, rate):
    # Mutasyon
    mutation_mask = np.random.rand(*child.shape) < rate
    mutation_values = np.random.uniform(-1, 1, size=child.shape)
    child += mutation_mask * mutation_values
    return child



def save_to_excel(objective_function, results):
    folder_name = f"{algorithm_name}_Results"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    df = pd.DataFrame(results, columns=["Solution", "Fitness"])
    file_path = os.path.join(folder_name, f"{objective_function.__name__}_results.xlsx")
    df.to_excel(file_path, index=False)

if __name__ == "__main__":
    dim = 30
    pop_size = 100
    num_generations = 1000
    mutation_rate = 0.01
    crossover_rate = 0.8

    benchmark_functions = [ackley, griewank, schwefel, rastrigin, sphere, perm, zakharov, rosenbrock, dixon_price]

    for objective_function in benchmark_functions:
        algorithm_name = "GeneticAlgorithm"
        results = []
        for _ in range(10):  # can change 10 to the desired number of runs for each function
            best_solution, best_fitness = genetic_algorithm(objective_function, pop_size, num_generations, mutation_rate, crossover_rate, dim)
            results.append((best_solution, best_fitness))
        save_to_excel(algorithm_name, objective_function, results)