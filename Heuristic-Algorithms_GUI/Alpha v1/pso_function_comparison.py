import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

class PSO:
    def __init__(self, num_particles, num_dimensions, max_iterations, objective_function):
        self.num_particles = num_particles
        self.num_dimensions = num_dimensions
        self.max_iterations = max_iterations
        self.particles = np.random.rand(num_particles, num_dimensions)
        self.velocities = np.zeros((num_particles, num_dimensions))
        self.global_best_position = np.zeros(num_dimensions)
        self.global_best_value = float('inf')
        self.objective_function = objective_function
        self.best_fitness_values = []

    def optimize(self):
        for iteration in range(self.max_iterations):
            for i in range(self.num_particles):
                current_position = self.particles[i]
                fitness = self.objective_function(current_position)

                if fitness < self.global_best_value:
                    self.global_best_value = fitness
                    self.global_best_position = current_position

                inertia_weight = 0.5
                cognitive_weight = 2.0
                social_weight = 2.0

                r1, r2 = np.random.rand(), np.random.rand()
                cognitive_component = cognitive_weight * r1 * (self.global_best_position - current_position)
                social_component = social_weight * r2 * (self.global_best_position - current_position)
                self.velocities[i] = inertia_weight * self.velocities[i] + cognitive_component + social_component
                self.particles[i] = current_position + self.velocities[i]

            self.best_fitness_values.append(self.global_best_value)

class PSOPlotter(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('PSO Fitness Comparison Plotter')
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        self.plot_button = QPushButton('Run PSO', self)
        self.plot_button.clicked.connect(self.run_pso)
        layout.addWidget(self.plot_button)

        self.canvas = PSOCanvas(self)
        layout.addWidget(self.canvas)

        self.setLayout(layout)

    def run_pso(self):
        num_particles = 30
        num_dimensions = 2
        max_iterations = 50

        ackley_pso = PSO(num_particles, num_dimensions, max_iterations, self.ackley_function)
        griewank_pso = PSO(num_particles, num_dimensions, max_iterations, self.griewank_function)
        schwefel_pso = PSO(num_particles, num_dimensions, max_iterations, self.schwefel_function)

        ackley_pso.optimize()
        griewank_pso.optimize()
        schwefel_pso.optimize()

        self.canvas.plot_data(ackley_pso.best_fitness_values, griewank_pso.best_fitness_values, schwefel_pso.best_fitness_values)

    @staticmethod
    def ackley_function(x):
        a = 20
        b = 0.2
        c = 2 * np.pi
        term1 = -a * np.exp(-b * np.sqrt(np.sum(x**2) / len(x)))
        term2 = -np.exp(np.sum(np.cos(c * x)) / len(x))
        return term1 + term2 + a + np.exp(1)

    @staticmethod
    def griewank_function(x):
        sum_term = np.sum(x**2) / 4000
        prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
        return 1 + sum_term - prod_term

    @staticmethod
    def schwefel_function(x):
        return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))

class PSOCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)

    def plot_data(self, ackley_values, griewank_values, schwefel_values):
        self.ax.clear()

        iterations = np.arange(1, len(ackley_values) + 1)

        self.ax.plot(iterations, ackley_values, label='Ackley', marker='o')
        self.ax.plot(iterations, griewank_values, label='Griewank', marker='o')
        self.ax.plot(iterations, schwefel_values, label='Schwefel', marker='o')

        self.ax.set_xlabel('Iterations')
        self.ax.set_ylabel('Fitness Value')
        self.ax.legend()
        self.fig.tight_layout()  # Add this line to ensure proper layout
        self.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PSOPlotter()
    window.show()
    sys.exit(app.exec_())
