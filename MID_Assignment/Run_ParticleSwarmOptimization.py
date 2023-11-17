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

# Particle Swarm Optimization Algorithm
def particle_swarm_optimization(objective_function, pop_size, num_generations, inertia_weight, cognitive_coeff, social_coeff, dim):
    positions = np.random.uniform(-32.768, 32.768, size=(pop_size, dim))
    velocities = np.random.uniform(-1, 1, size=(pop_size, dim))

    personal_best_positions = np.copy(positions)
    personal_best_fitness = np.array([objective_function(ind) for ind in positions])

    global_best_index = np.argmin(personal_best_fitness)
    global_best_position = np.copy(personal_best_positions[global_best_index])
    global_best_fitness = personal_best_fitness[global_best_index]

    for generation in range(num_generations):
        for i in range(pop_size):
            # Update velocity
            inertia_term = inertia_weight * velocities[i]
            cognitive_term = cognitive_coeff * np.random.random() * (personal_best_positions[i] - positions[i])
            social_term = social_coeff * np.random.random() * (global_best_position - positions[i])

            velocities[i] = inertia_term + cognitive_term + social_term

            # Update position
            positions[i] += velocities[i]

            # Clip positions to be within bounds
            positions[i] = np.clip(positions[i], -32.768, 32.768)

            # Update personal best
            current_fitness = objective_function(positions[i])
            if current_fitness < personal_best_fitness[i]:
                personal_best_fitness[i] = current_fitness
                personal_best_positions[i] = np.copy(positions[i])

                # Update global best
                if current_fitness < global_best_fitness:
                    global_best_fitness = current_fitness
                    global_best_position = np.copy(positions[i])

    return global_best_position, global_best_fitness

def save_to_excel(algorithm_name, objective_function, results):
    folder_name = f"{algorithm_name}_Results"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    df = pd.DataFrame(results, columns=["Solution", "Fitness"])
    file_path = os.path.join(folder_name, f"{objective_function.__name__}_results.xlsx")
    df.to_excel(file_path, index=False)

# Example usage for PSO on each benchmark function
if __name__ == "__main__":
    dim = 30
    pop_size = 50
    num_generations = 100
    inertia_weight = 0.5
    cognitive_coeff = 2.0
    social_coeff = 2.0

    benchmark_functions = [ackley, griewank, schwefel, rastrigin, sphere, perm, zakharov, rosenbrock, dixon_price]

   
        
    for objective_function in benchmark_functions:
        algorithm_name = "ParticleSwarmOptimization"
        results = []
        for _ in range(10):  # You can change 10 to the desired number of runs for each function
            best_solution, best_fitness = particle_swarm_optimization(objective_function, pop_size, num_generations, inertia_weight, cognitive_coeff, social_coeff, dim)
            results.append((best_solution, best_fitness))
        save_to_excel(algorithm_name, objective_function, results)
