
from ui_mainwindow import Ui_MainWindow
from PyQt5.QtGui import QPixmap
from io import BytesIO
from pyqtgraph import PlotWidget
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QGraphicsScene, QGraphicsPixmapItem
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from PSO import PSO
from SA import SA_geometric, SA_linear
from GWO import GWO
from HS import HS
from GA import GA

import functions
from enumFunctions import Functions
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from solution import solution

bounds = [
    [-32.768, 32768], [-600, 600], [-500, 500], [-5.12, 5.12], [-5.12, 5.12],
    [-30.30], [-5.10], [-2048, 2048], [-10, 10]
]

class MatplotlibWidget(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig, self.axes = plt.subplots(figsize=(width, height), dpi=dpi)
        super(MatplotlibWidget, self).__init__(fig)
        self.setParent(parent)

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
        self.sol = solution()
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
    # Run PSO
        if self.algorithm_comboBox.currentText() == "PSO":
            self.run_algorithm(self.algorithm_comboBox, "PSO")

    # Run SA
        if self.algorithm_comboBox.currentText() == "SA":
            self.run_algorithm(self.algorithm_comboBox, "SA")

    # Run GWO
        if self.algorithm_comboBox.currentText() == "GWO":
            self.run_algorithm(self.algorithm_comboBox, "GWO")

    # Run PSO for combo box 2
        if self.algorithm_comboBox_2.currentText() == "PSO":
            self.run_algorithm(self.algorithm_comboBox_2, "PSO")

    # Run SA for combo box 2
        if self.algorithm_comboBox_2.currentText() == "SA":
            self.run_algorithm(self.algorithm_comboBox_2, "SA")

    # Run GWO for combo box 2
        if self.algorithm_comboBox_2.currentText() == "GWO":
            self.run_algorithm(self.algorithm_comboBox_2, "GWO")

    # Run PSO for combo box 3
        if self.algorithm_comboBox_3.currentText() == "PSO":
            self.run_algorithm(self.algorithm_comboBox_3, "PSO")

    # Run SA for combo box 3
        if self.algorithm_comboBox_3.currentText() == "SA":
            self.run_algorithm(self.algorithm_comboBox_3, "SA")

    # Run GWO for combo box 3
        if self.algorithm_comboBox_3.currentText() == "GWO":
            self.run_algorithm(self.algorithm_comboBox_3, "GWO")

def run_algorithm(self, combo_box, algorithm_name):
    selected_index = self.func_comboBox.currentIndex()
    dim = 30
    obj_func = self.function_select()
    lower_bound = bounds[selected_index][0]
    upper_bound = bounds[selected_index][1]

    if algorithm_name == "PSO":
        pop_size = int(self.pso_pop_size.text())
        num_gen = int(self.pso_num_gen.text())
        algorithm_sol = PSO(obj_func, lower_bound, upper_bound, dim, pop_size, num_gen)
    elif algorithm_name == "SA":
        temp = int(self.sa_temp_2.text())
        sa_type = self.SA_type_2.currentText()
        lower_bounds = [None for _ in range(dim)]
        upper_bounds = [None for _ in range(dim)]
        for idx in range(dim):
            lower_bounds[idx] = bounds[selected_index][0]
            upper_bounds[idx] = bounds[selected_index][1]

        if sa_type == "Linear":
            algorithm_sol = SA_linear(dim=dim, min_values=lower_bounds, max_values=upper_bounds, mu=0, sigma=1,
                                      initial_temperature=temp, temperature_iterations=5000,
                                      final_temperature=0.0001, alpha=0.95, target_function=obj_func, verbose=True)
        else:
            algorithm_sol = SA_geometric(dim=dim, min_values=lower_bounds, max_values=upper_bounds, mu=0, sigma=1,
                                         initial_temperature=temp, temperature_iterations=5000,
                                         final_temperature=0.0001, alpha=0.98, target_function=obj_func, verbose=True)
    elif algorithm_name == "GWO":
        search_agents_no = int(self.GWO_SearchAgentsNo_3.text())
        max_iter = int(self.GWO_maxIter_3.text())
        decrease_from = int(self.GWO_decreaseFrom_3.text())

        algorithm_sol = GWO(obj_func, lower_bound, upper_bound, dim, search_agents_no, max_iter, decrease_from)

    self.algorithm_comboBox_type = 1 if algorithm_name == "PSO" else 2 if algorithm_name == "SA" else 3
    self.plot_algorithm_result(algorithm_sol, algorithm_name)


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

    def plot_algorithm_result(self, algorithm_sol, algorithm_name):
        existing_legend = self.matplotlibWidget.axes.get_legend()

        if self.algorithm_comboBox_type == 1:
            color = "red"
        elif self.algorithm_comboBox_type == 2:
            color = "blue"
        elif self.algorithm_comboBox_type == 3:
            color = "green"

        line = sns.lineplot(x=algorithm_sol.x, y=algorithm_sol.y, ax=self.matplotlibWidget.axes, color=color,
                            label=algorithm_name)

        best_fitness_iteration = np.argmin(algorithm_sol.y)
        best_fitness_value = np.min(algorithm_sol.y)
        self.matplotlibWidget.axes.plot(best_fitness_iteration + 1, best_fitness_value, 'ro',
                                        label=f'Best Fitness ({algorithm_name})', markersize=8)

        self.matplotlibWidget.axes.set_xlabel('Iteration count')
        self.matplotlibWidget.axes.set_ylabel('Fitness value')

        if existing_legend:
            self.matplotlibWidget.axes.legend(handles=existing_legend.legendHandles, loc=(1, 0.7))
        else:
            self.matplotlibWidget.axes.legend(loc=(1, 0.7))

        self.matplotlibWidget.draw()

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
