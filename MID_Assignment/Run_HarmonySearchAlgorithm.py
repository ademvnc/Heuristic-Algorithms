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

# Harmony Search Algorithm
def harmony_search(objective_function, harmony_memory_size, num_iterations, harmony_memory_rate, pitch_adjust_rate, dim):
    # Initialize harmony memory
    harmony_memory = np.random.uniform(-32.768, 32.768, size=(harmony_memory_size, dim))
    harmony_fitness = np.array([objective_function(ind) for ind in harmony_memory])

    for iteration in range(num_iterations):
        # Create new harmony
        new_harmony = create_harmony(harmony_memory, harmony_memory_rate, pitch_adjust_rate)

        # Evaluate the fitness of the new harmony
        new_fitness = objective_function(new_harmony)

        # Update harmony memory if the new harmony is better
        min_index = np.argmin(harmony_fitness)
        if new_fitness < harmony_fitness[min_index]:
            harmony_memory[min_index] = new_harmony
            harmony_fitness[min_index] = new_fitness

    # Return the best solution found
    best_index = np.argmin(harmony_fitness)
    best_solution = harmony_memory[best_index]

    return best_solution, harmony_fitness[best_index]

def create_harmony(harmony_memory, harmony_memory_rate, pitch_adjust_rate):
    # Randomly choose one or more harmonies from memory
    num_harmonies = int(np.ceil(harmony_memory_rate * len(harmony_memory)))
    selected_harmonies = harmony_memory[np.random.choice(len(harmony_memory), num_harmonies, replace=False)]

    # Adjust the pitch of the selected harmonies
    pitch_adjusted_harmonies = adjust_pitch(selected_harmonies, pitch_adjust_rate)

    # Combine the original and pitch-adjusted harmonies
    new_harmony = np.vstack((selected_harmonies, pitch_adjusted_harmonies))

    # Randomly select one element for each dimension from the new harmony
    result_harmony = np.array([np.random.choice(new_harmony[:, i]) for i in range(new_harmony.shape[1])])

    return result_harmony

def adjust_pitch(harmonies, pitch_adjust_rate):
    # Adjust the pitch of the harmonies
    pitch_adjusted_harmonies = harmonies + pitch_adjust_rate * np.random.uniform(-1, 1, size=harmonies.shape)

    # Clip values to be within bounds
    pitch_adjusted_harmonies = np.clip(pitch_adjusted_harmonies, -32.768, 32.768)

    return pitch_adjusted_harmonies

def save_to_excel(algorithm_name, objective_function, results):
    folder_name = f"{algorithm_name}_Results"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    df = pd.DataFrame(results, columns=["Solution", "Fitness"])
    file_path = os.path.join(folder_name, f"{objective_function.__name__}_results.xlsx")
    df.to_excel(file_path, index=False)

# Example usage for Harmony Search on each benchmark function
if __name__ == "__main__":
    dim = 30
    harmony_memory_size = 20
    num_iterations = 100
    harmony_memory_rate = 0.7
    pitch_adjust_rate = 0.01

    benchmark_functions = [ackley, griewank, schwefel, rastrigin, sphere, perm, zakharov, rosenbrock, dixon_price]


    for objective_function in benchmark_functions:
        algorithm_name = "HarmonySearch"
        results = []
        for _ in range(10):  # You can change 10 to the desired number of runs for each function
            best_solution, best_fitness = harmony_search(objective_function, harmony_memory_size, num_iterations, harmony_memory_rate, pitch_adjust_rate, dim)
            results.append((best_solution, best_fitness))
        save_to_excel(algorithm_name, objective_function, results)
