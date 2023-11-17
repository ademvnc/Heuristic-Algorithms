import os
import numpy as np
import pandas as pd

# Benchmark functions and their bounds
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

# Grey Wolf Optimization Algorithm
def grey_wolf_optimization(objective_function, pop_size, num_generations, a_values, dim):
    positions = np.random.uniform(-32.768, 32.768, size=(pop_size, dim))

    for generation in range(num_generations):
        a = a_values[min(generation, len(a_values) - 1)]
        alpha, beta, delta = get_alpha_beta_delta(positions, objective_function)

        for i in range(pop_size):
            for j in range(dim):
                rand_1 = np.random.random()
                rand_2 = np.random.random()

                A1 = 2 * a * rand_1 - a
                C1 = 2 * rand_2

                D_alpha = np.abs(C1 * alpha[j] - positions[i, j])
                X1 = alpha[j] - A1 * D_alpha

                rand_1 = np.random.random()
                rand_2 = np.random.random()

                A2 = 2 * a * rand_1 - a
                C2 = 2 * rand_2

                D_beta = np.abs(C2 * beta[j] - positions[i, j])
                X2 = beta[j] - A2 * D_beta

                rand_1 = np.random.random()
                rand_2 = np.random.random()

                A3 = 2 * a * rand_1 - a
                C3 = 2 * rand_2

                D_delta = np.abs(C3 * delta[j] - positions[i, j])
                X3 = delta[j] - A3 * D_delta

                positions[i, j] = (X1 + X2 + X3) / 3.0

    # Return the best solution found
    best_index = np.argmin([objective_function(ind) for ind in positions])
    best_solution = positions[best_index]

    return best_solution, objective_function(best_solution)

def get_alpha_beta_delta(positions, objective_function):
    fitness_values = [objective_function(ind) for ind in positions]
    sorted_indices = np.argsort(fitness_values)

    alpha = positions[sorted_indices[0]]
    beta = positions[sorted_indices[1]]
    delta = positions[sorted_indices[2]]

    return alpha, beta, delta

def save_to_excel(algorithm_name, objective_function, results):
    folder_name = f"{algorithm_name}_Results"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    df = pd.DataFrame(results, columns=["Solution", "Fitness"])
    file_path = os.path.join(folder_name, f"{objective_function.__name__}_results.xlsx")
    df.to_excel(file_path, index=False)

# Example usage for GWO on each benchmark function
if __name__ == "__main__":
    dim = 30
    pop_size = 50
    num_generations = 100
    a_values = [4, 3, 2]

    benchmark_functions = [ackley, griewank, schwefel, rastrigin, sphere, perm, zakharov, rosenbrock, dixon_price]

    for objective_function in benchmark_functions:
        algorithm_name = "GWO"
        results = []
        for _ in range(10):  # You can change 10 to the desired number of runs for each function
            best_solution, best_fitness = grey_wolf_optimization(objective_function, pop_size, num_generations, a_values, dim)
            results.append((best_solution, best_fitness))
        save_to_excel(algorithm_name, objective_function, results)