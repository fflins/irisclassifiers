from sklearn.metrics import confusion_matrix
import seaborn as sns
import tkinter as tk
from tkinter import ttk
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import pandas as pd

def show_confusion_matrix_popup(self, expected, predictions):
        popup = tk.Toplevel(self.root)
        popup.title("Matriz de Confusão")

        # frame para a matriz de confusão
        cm_frame = ttk.Frame(popup)
        cm_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        cm = confusion_matrix(expected, predictions)
        classes = np.unique(np.concatenate((expected, predictions)))

        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)

        # transpor a matriz de confusão pra ficar igual o slide
        sns.heatmap(cm.T, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=classes, yticklabels=classes)

        # inverte os rótulos 
        ax.set_xlabel('Valores Reais')
        ax.set_ylabel('Previsões')
        ax.set_title('Matriz de Confusão')

        canvas = FigureCanvasTkAgg(fig, master=cm_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


#concluir
def visualize_perceptron_decision_surface(self, weights, class1, class2):
    # Cria nova janela
    popup = tk.Toplevel(self.root)
    popup.title(f"Superfície de Decisão do Perceptron: {class1} vs {class2}")
    popup.geometry("800x600")
    
    # Carrega os dados
    data = pd.read_csv("../data.csv", decimal=",")
    
    # Frame para controles
    control_frame = ttk.Frame(popup)
    control_frame.pack(fill=tk.X, pady=5)
    
    # Seleção de características para visualização
    ttk.Label(control_frame, text="Característica X:").pack(side=tk.LEFT, padx=5)
    features = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width']
    feature_x = tk.StringVar(value=features[0])
    feature_x_combo = ttk.Combobox(control_frame, textvariable=feature_x, values=features, state="readonly", width=15)
    feature_x_combo.pack(side=tk.LEFT, padx=5)
    
    ttk.Label(control_frame, text="Característica Y:").pack(side=tk.LEFT, padx=5)
    feature_y = tk.StringVar(value=features[2])
    feature_y_combo = ttk.Combobox(control_frame, textvariable=feature_y, values=features, state="readonly", width=15)
    feature_y_combo.pack(side=tk.LEFT, padx=5)
    
    # Frame para o gráfico
    plot_frame = ttk.Frame(popup)
    plot_frame.pack(fill=tk.BOTH, expand=True, pady=5)
    
    # Função para atualizar o gráfico
    def update_plot(*args):
        return


