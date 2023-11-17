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

# Simulated Annealing
def simulated_annealing(objective_function, initial_solution, initial_temperature, cooling_factor, num_iterations):
    current_solution = initial_solution
    current_fitness = objective_function(current_solution)
    best_solution = current_solution
    best_fitness = current_fitness
    temperature = initial_temperature

    for iteration in range(num_iterations):
        # Generate a neighboring solution
        neighbor_solution = generate_neighbor(current_solution)

        # Calculate fitness for the neighbor
        neighbor_fitness = objective_function(neighbor_solution)

        # Decide whether to accept the neighbor
        if neighbor_fitness < current_fitness or np.random.rand() < acceptance_probability(current_fitness, neighbor_fitness, temperature):
            current_solution = neighbor_solution
            current_fitness = neighbor_fitness

        # Update the best solution if needed
        if current_fitness < best_fitness:
            best_solution = current_solution
            best_fitness = current_fitness

        # Cool the temperature
        temperature *= cooling_factor

    return best_solution, best_fitness

def generate_neighbor(solution):
    # Generate a neighboring solution by perturbing the current solution
    perturbation = np.random.uniform(-0.1, 0.1, size=solution.shape)
    neighbor_solution = solution + perturbation
    return neighbor_solution

def acceptance_probability(current_fitness, neighbor_fitness, temperature):
    # Probability of accepting a worse solution
    return np.exp(-(neighbor_fitness - current_fitness) / temperature)

def save_to_excel(algorithm_name, objective_function, results):
    folder_name = f"{algorithm_name}_Results"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    df = pd.DataFrame(results, columns=["Solution", "Fitness"])
    file_path = os.path.join(folder_name, f"{objective_function.__name__}_results.xlsx")
    df.to_excel(file_path, index=False)

# Simulated Annealing'in kullanımı
if __name__ == "__main__":
    dim = 30
    initial_solution = np.random.uniform(-32.768, 32.768, size=dim)
    initial_temperature = 1000
    cooling_factor = 0.99
    num_iterations = 1000

    benchmark_functions = [ackley, griewank, schwefel, rastrigin, sphere, perm, zakharov, rosenbrock, dixon_price]

    

        
    for objective_function in benchmark_functions:
        algorithm_name = "SimulatedAnnealing"
        results = []
        for _ in range(10):  # You can change 10 to the desired number of runs for each function
            best_solution, best_fitness = simulated_annealing(objective_function, initial_solution, initial_temperature, cooling_factor, num_iterations)
            results.append((best_solution, best_fitness))
        save_to_excel(algorithm_name, objective_function, results)