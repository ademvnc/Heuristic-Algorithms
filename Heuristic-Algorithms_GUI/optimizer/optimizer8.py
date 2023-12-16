from HS import HS
from GWO import GWO
from PSO import PSO
from GA import GA
from SA import SA_geometric, SA_linear
from ui_mainwindow import Ui_MainWindow
from PyQt5.QtWidgets import QApplication, QMainWindow
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import numpy as np
import functions
from enumFunctions import Functions

# bounds değişkenini tanımlayın
bounds = [
    [-32.768, 32768], [-600, 600], [-500, 500], [-5.12, 5.12], [-5.12, 5.12],
    [-30.30], [-5.10], [-2048, 2048], [-10, 10]
]

class MatplotlibWidget(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig, self.axes = plt.subplots(figsize=(width, height), dpi=dpi)
        super(MatplotlibWidget, self).__init__(fig)
        self.setParent(parent)

# ... (Kodun geri kalanı)

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)

        self.matplotlibWidget = MatplotlibWidget(self.graphWidget)
        self.graphLayout.addWidget(self.matplotlibWidget)

        self.color_legend = {}
        self.stackedWidget.setCurrentIndex(0)
        self.stackedWidget_2.setCurrentIndex(0)
        self.stackedWidget_3.setCurrentIndex(0)
        self.algorithm_comboBox.currentIndexChanged.connect(self.toggle_layer)
        self.algorithm_comboBox_2.currentIndexChanged.connect(self.toggle_layer)
        self.algorithm_comboBox_3.currentIndexChanged.connect(self.toggle_layer)
        self.Run_button.clicked.connect(self.button_clicked)
        self.sol = None
        self.algorithm_comboBox_type = 0
        self.clearGraphButton.clicked.connect(self.clear_graph)

    def toggle_layer(self):
        self.stackedWidget.setCurrentIndex(self.algorithm_comboBox.currentIndex())
        self.stackedWidget_2.setCurrentIndex(self.algorithm_comboBox_2.currentIndex())
        self.stackedWidget_3.setCurrentIndex(self.algorithm_comboBox_3.currentIndex())

    def clear_graph(self):
        # Clear the graph in the MatplotlibWidget
        self.matplotlibWidget.axes.clear()
        self.matplotlibWidget.draw()

    def button_clicked(self):
        if self.algorithm_comboBox.currentText() == "PSO":
            selected_index = self.algorithm_comboBox.currentIndex()
            dim = 30
            pop_size = int(self.pso_pop_size.text())
            num_gen = int(self.pso_num_gen.text())
            obj_func = self.function_select()
            lower_bound = bounds[selected_index][0]
            upper_bound = bounds[selected_index][1]
            self.sol = PSO(obj_func, lower_bound, upper_bound, dim, pop_size, num_gen)
            self.algorithm_comboBox_type = 1
            self.run_algorithm_iterations(num_gen)

        elif self.algorithm_comboBox.currentText() == "SA":
            selected_index = self.func_comboBox.currentIndex()
            dim = 30
            temp = int(self.sa_temp_2.text())
            sa_type = self.SA_type_2.currentText()
            obj_func = self.function_select()
            lower_bounds = [None for _ in range(dim)]
            upper_bounds = [None for _ in range(dim)]
            for idx in range(dim):
                lower_bounds[idx] = bounds[selected_index][0]
                upper_bounds[idx] = bounds[selected_index][1]

            if sa_type == "Linear":
                self.sol = SA_linear(dim=dim, min_values=lower_bounds, max_values=upper_bounds, mu=0, sigma=1,
                                     initial_temperature=temp, temperature_iterations=5000,
                                     final_temperature=0.0001, alpha=0.9, target_function=obj_func, verbose=True)
            else:
                self.sol = SA_geometric(dim=dim, min_values=lower_bounds, max_values=upper_bounds, mu=0, sigma=1,
                                        initial_temperature=temp, temperature_iterations=5000,
                                        final_temperature=0.0001, alpha=0.9, target_function=obj_func, verbose=True)

            self.algorithm_comboBox_type = 2
            self.run_algorithm_iterations(5000)

        elif self.algorithm_comboBox.currentText() == "GWO":
            selected_index = self.algorithm_comboBox.currentIndex()
            dim = 30
            pop_size = int(self.pso_pop_size.text())
            num_gen = int(self.pso_num_gen.text())
            obj_func = self.function_select()
            lower_bound = bounds[selected_index][0]
            upper_bound = bounds[selected_index][1]
            search_agents_no = int(self.GWO_SearchAgentsNo_3.text())
            max_iter = int(self.GWO_maxIter_3.text())
            decrease_from = int(self.GWO_decreaseFrom_3.text())

            self.sol = GWO(obj_func, lower_bound, upper_bound, dim, search_agents_no, max_iter, decrease_from)
            self.algorithm_comboBox_type = 3
            self.run_algorithm_iterations(num_gen)

        # Diğer algoritmalar için benzer değişiklikler...

    def run_algorithm_iterations(self, num_iterations):
        for i in range(num_iterations):
            # Algoritmanın iterasyonlarını çalıştır
            self.sol.run_iteration()

            # Her iterasyon sonunda en iyi fitness değerini ve iterasyon numarasını al
            iteration_number = i + 1
            best_fitness_value = self.sol.get_best_fitness()

            # En iyi fitness değerini ve iterasyon numarasını grafik üzerine çiz
            self.plot_best_fitness(iteration_number, best_fitness_value)

        self.update_graph()

    def function_select(self):
        func = self.func_comboBox.currentIndex()
        if func == 0:
            return functions.selectFunction(Functions.ackley)
        elif func == 1:
            return functions.selectFunction(Functions.griewank)
        elif func == 2:
            return functions.selectFunction(Functions.schwefel)
        elif func == 3:
            return functions.selectFunction(Functions.rastrigin)
        elif func == 4:
            return functions.selectFunction(Functions.sphere)
        elif func == 5:
            return functions.selectFunction(Functions.perm)
        elif func == 6:
            return functions.selectFunction(Functions.zakharov)
        elif func == 7:
            return functions.selectFunction(Functions.rosenbrock)
        elif func == 8:
            return functions.selectFunction(Functions.dixonprice)

    def plot_best_fitness(self, iteration, fitness):
        self.matplotlibWidget.axes.plot(iteration, fitness, 'ro', label='Best Fitness', markersize=8)
        self.matplotlibWidget.draw()

    def update_graph(self):
        line = sns.lineplot(x=self.sol.x, y=self.sol.y, ax=self.matplotlibWidget.axes)

        self.matplotlibWidget.axes.set_xlabel('Iteration count')
        self.matplotlibWidget.axes.set_ylabel('Fitness value')
        existing_legend = self.matplotlibWidget.axes.get_legend()

        algorithm_names = ["PSO", "SA", "GWO", "GA", "HS"]
        algorithm_colors = ["red", "blue", "green", "purple", "orange"]

        algorithm_name = algorithm_names[self.algorithm_comboBox_type - 1]
        algorithm_color = algorithm_colors[self.algorithm_comboBox_type - 1]

        line.lines[0].set_color(algorithm_color)
        legend_color = mpatches.Patch(color=algorithm_color, label=f"{algorithm_name}")

        if existing_legend:
            self.matplotlibWidget.axes.legend(handles=existing_legend.legendHandles + [legend_color], loc=(1, 0.7))
        else:
            self.matplotlibWidget.axes.legend(handles=[legend_color], loc=(1, 0.7))

        # Best fitness değerlerini göstermek için
        best_fitness_iteration = np.argmin(self.sol.y)
        best_fitness_value = np.min(self.sol.y)
        self.plot_best_fitness(best_fitness_iteration + 1, best_fitness_value)

        self.matplotlibWidget.draw()

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
