import functions
from enumFunctions import Functions
from SA import simulated_annealing

def SimulatedAnnealing():
    obj_func = functions.selectFunction(Functions.griewank)
    # dim array size, -5 lb +5 lb 
    simulated_annealing( min_values = [-600,-600,-600,-600,-600,-600,-600,-600,-600,-600], max_values = [600,600,600,600,600,600,600,600,600,600], mu = 0, sigma = 1, initial_temperature = 1.0, temperature_iterations = 100,
        final_temperature = 0.0001, alpha = 0.9, target_function = obj_func, verbose = True)

def main():
    SimulatedAnnealing()

if __name__ == "__main__":
    main()