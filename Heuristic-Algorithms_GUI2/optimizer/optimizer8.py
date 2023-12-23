import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QComboBox, QLineEdit, QPushButton
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from scipy.optimize import minimize

class OptimizationAlgorithm:
    def __init__(self, name, function, parameters):
        self.name = name
        self.function = function
        self.parameters = parameters

    def run(self):
        # Algoritma çalıştırma işlemleri burada gerçekleştirilir
        result = self.function(self.parameters)
        return result

class FunctionSelector(QWidget):
    def __init__(self):
        super().__init__()

        self.algorithms = ["PSO", "GWO", "HS", "GA", "SA"]
        self.functions = ["ackley", "griewank", "schwefel", "rastrigin", "sphere", "perm", "zakharov", "rosenbrock", "dixonprice"]

        self.initUI()

    def initUI(self):
        self.algorithmSelector = QComboBox(self)
        self.algorithmSelector.addItems(self.algorithms)

        self.functionSelector = QComboBox(self)
        self.functionSelector.addItems(self.functions)

        self.parametersInput = QLineEdit(self)

        self.runButton = QPushButton('Run Algorithm', self)
        self.runButton.clicked.connect(self.runAlgorithm)

        self.canvas = PlotCanvas(self)

        layout = QVBoxLayout()
        layout.addWidget(QLabel('Select Algorithm:'))
        layout.addWidget(self.algorithmSelector)
        layout.addWidget(QLabel('Select Function:'))
        layout.addWidget(self.functionSelector)
        layout.addWidget(QLabel('Enter Parameters (comma-separated):'))
        layout.addWidget(self.parametersInput)
        layout.addWidget(self.runButton)
        layout.addWidget(self.canvas)

        self.setLayout(layout)

    def runAlgorithm(self):
        algorithm = self.algorithmSelector.currentText()
        function = self.functionSelector.currentText()
        parameters = self.parametersInput.text()

        # Algoritma ve fonksiyonlara göre işlemleri gerçekleştirin
        result = self.runOptimizationAlgorithm(algorithm, function, parameters)

        # Grafik çizimini sağlayın
        self.canvas.plotResult(result)

    def runOptimizationAlgorithm(self, algorithm, function, parameters):
        # Seçilen algoritma ve fonksiyonun parametrelerini ayarlayın
        algorithm_params = {key: int(val) for key, val in [param.split(':') for param in parameters.split(',')]}

        # Algoritmayı başlat
        opt_algorithm = OptimizationAlgorithm(algorithm, self.getFunctionByName(function), algorithm_params)
        result = opt_algorithm.run()

        return {'result': result, 'algorithm': algorithm, 'function': function}

    def getFunctionByName(self, name):
        # Fonksiyon ismine göre gerçek fonksiyonu seçin
        functions = {
            'ackley': self.ackley,
            'griewank': self.griewank,
            'schwefel': self.schwefel,
            'rastrigin': self.rastrigin,
            'sphere': self.sphere,
            'perm': self.perm,
            'zakharov': self.zakharov,
            'rosenbrock': self.rosenbrock,
            'dixonprice': self.dixonprice,
        }
        return functions[name]

    def ackley(self, params):
        # Ackley fonksiyonu
        # Burada algoritmanızın gerçek çalışma mantığını uygulayın
        # Şu an sadece rasgele bir değer döndürüyoruz
        return np.random.rand()

    def griewank(self, params):
        # Griewank fonksiyonu
        return np.random.rand()

    def schwefel(self, params):
        # Schwefel fonksiyonu
        return np.random.rand()

    def rastrigin(self, params):
        # Rastrigin fonksiyonu
        return np.random.rand()

    def sphere(self, params):
        # Sphere fonksiyonu
        return np.random.rand()

    def perm(self, params):
        # Perm fonksiyonu
        return np.random.rand()

    def zakharov(self, params):
        # Zakharov fonksiyonu
        return np.random.rand()

    def rosenbrock(self, params):
        # Rosenbrock fonksiyonu
        return np.random.rand()

    def dixonprice(self, params):
        # Dixon-Price fonksiyonu
        return np.random.rand()

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self, QWidget.QSizePolicy.Expanding, QWidget.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def plotResult(self, result):
        algorithm = result['algorithm']
        function = result['function']
        fitness_value = result['result']

        # Grafik çizimini sağlayın
        self.axes.plot(algorithm, fitness_value, label=f'{algorithm} - {function}')
        self.axes.scatter(algorithm, fitness_value, color='red')
        self.axes.set_xlabel('Iteration')
        self.axes.set_ylabel('Fitness Value')
        self.axes.legend()
        self.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = FunctionSelector()
    mainWindow.setGeometry(100, 100, 800, 600)
    mainWindow.show()
    sys.exit(app.exec_())
