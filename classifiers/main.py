import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd
from classifiers import get_decision_boundary_equation
from data_loader import load_data
import numpy as np

class IrisClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Visualização de Classificadores")
        self.root.geometry("1000x700")

        # Carregar dados corretamente
        (self.data, self.training_sample, self.test_sample, 
         self.setosa_mean, self.versicolor_mean, self.virginica_mean) = load_data()
        
        # Calcular média combinada APENAS com dados de TREINAMENTO
        non_setosa_train = self.training_sample[
            ~self.training_sample['Species'].isin(['setosa'])
        ]
        self.merged_mean = non_setosa_train.drop(columns='Species').mean().values

        # Definir nomes das características PRIMEIRO
        self.feature_names = ['Comprimento da Sépala', 'Largura da Sépala',
                              'Comprimento da Pétala', 'Largura da Pétala']

        # Configuração da interface
        self.create_control_panel()
        self.create_plot_area()

        # Índices padrão das características
        self.feature_x = 0
        self.feature_y = 1

    def create_control_panel(self):
        control_frame = ttk.LabelFrame(self.root, text="Controles")
        control_frame.pack(fill="x", padx=10, pady=10)

        # Combobox para seleção de características
        ttk.Label(control_frame, text="Característica X:").grid(row=0, column=0, padx=5)
        self.x_var = tk.StringVar(value=self.feature_names[0])
        ttk.Combobox(control_frame, textvariable=self.x_var, 
                    values=self.feature_names, state="readonly").grid(row=0, column=1)

        ttk.Label(control_frame, text="Característica Y:").grid(row=0, column=2, padx=5)
        self.y_var = tk.StringVar(value=self.feature_names[1])
        ttk.Combobox(control_frame, textvariable=self.y_var, 
                    values=self.feature_names, state="readonly").grid(row=0, column=3)

        # Botões de ação
        btn_frame = ttk.Frame(control_frame)
        btn_frame.grid(row=1, column=0, columnspan=4, pady=10)
        
        ttk.Button(btn_frame, text="Mostrar Dados Originais", 
                  command=self.plot_original).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Mostrar Visão Binária", 
                  command=self.plot_binary).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Classificador Euclidiano", 
                  command=lambda: self.plot_classifier("euclidean")).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Classificador LDA", 
                  command=lambda: self.plot_classifier("lda")).pack(side=tk.LEFT, padx=5)

    def create_plot_area(self):
        self.figure = Figure(figsize=(8, 5))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
    def get_selected_features(self):
        self.feature_x = self.feature_names.index(self.x_var.get())
        self.feature_y = self.feature_names.index(self.y_var.get())

    def plot_original(self):
        self.get_selected_features()
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Plotar TODAS as classes originalmente
        for species, color in [('setosa', 'red'), ('versicolor', 'blue'), 
                              ('virginica', 'green')]:
            subset = self.data[self.data['Species'] == species]
            ax.scatter(subset.iloc[:, self.feature_x], 
                      subset.iloc[:, self.feature_y], 
                      c=color, label=species, alpha=0.7)
        
        # Plotar centroides do TREINAMENTO
        ax.scatter(self.setosa_mean[self.feature_x], self.setosa_mean[self.feature_y], 
                  c='black', marker='x', s=200, label='Centroides')
        ax.scatter(self.versicolor_mean[self.feature_x], self.versicolor_mean[self.feature_y], 
                  c='black', marker='x', s=200)
        ax.scatter(self.virginica_mean[self.feature_x], self.virginica_mean[self.feature_y], 
                  c='black', marker='x', s=200)
        
        ax.set_title("Dados Originais com Centroides")
        ax.set_xlabel(self.x_var.get())
        ax.set_ylabel(self.y_var.get())
        ax.legend()
        self.canvas.draw()

    def plot_binary(self):
        self.get_selected_features()
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Separar classes usando TODOS os dados (mantendo versicolor e virginica distintos)
        setosa = self.data[self.data['Species'] == 'setosa']
        non_setosa = self.data[self.data['Species'].isin(['versicolor', 'virginica'])]
        
        # Plotar dados
        ax.scatter(setosa.iloc[:, self.feature_x], setosa.iloc[:, self.feature_y], 
                  c='red', label='Setosa', alpha=0.7)
        ax.scatter(non_setosa.iloc[:, self.feature_x], non_setosa.iloc[:, self.feature_y], 
                  c='blue', label='Não-Setosa', alpha=0.7)
        
        # Plotar centroides do TREINAMENTO
        ax.scatter(self.setosa_mean[self.feature_x], self.setosa_mean[self.feature_y], 
                  c='black', marker='x', s=200, label='Centroides')
        ax.scatter(self.merged_mean[self.feature_x], self.merged_mean[self.feature_y], 
                  c='black', marker='x', s=200)
        
        ax.set_title("Visualização Binária (Setosa vs Outras)")
        ax.set_xlabel(self.x_var.get())
        ax.set_ylabel(self.y_var.get())
        ax.legend()
        self.canvas.draw()

    def plot_classifier(self, classifier_type):
        self.get_selected_features()
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Plotar dados binários (TODOS os dados)
        setosa = self.data[self.data['Species'] == 'setosa']
        non_setosa = self.data[~self.data['Species'].isin(['setosa'])]
        
        ax.scatter(setosa.iloc[:, self.feature_x], setosa.iloc[:, self.feature_y], 
                  c='red', label='Setosa', alpha=0.7)
        ax.scatter(non_setosa.iloc[:, self.feature_x], non_setosa.iloc[:, self.feature_y], 
                  c='blue', label='Não-Setosa', alpha=0.7)
        
        # Obter médias das características SELECIONADAS (do treinamento)
        setosa_mean = self.setosa_mean[[self.feature_x, self.feature_y]]
        merged_mean = self.merged_mean[[self.feature_x, self.feature_y]]
        
        # Calcular fronteira de decisão
        if classifier_type == "euclidean":
            a, b = get_decision_boundary_equation(setosa_mean, merged_mean)
            title = "Classificador de Distância Euclidiana"
        else:
            a, b = get_decision_boundary_equation(merged_mean, setosa_mean)
            a, b = -a, -b  # Inverter para LDA
            title = "Classificador LDA"
        
        # Gerar linha de decisão
        x_vals = np.array(ax.get_xlim())
        if a[1] != 0:
            y_vals = (-a[0]*x_vals - b)/a[1]
        else:
            y_vals = np.full_like(x_vals, -b/a[0]) if a[0] != 0 else [0, 0]
        
        ax.plot(x_vals, y_vals, 'k--', lw=2, label='Fronteira de Decisão')
        
        # Exibir equação
        equation = f"{a[0]:.2f}x + {a[1]:.2f}y + {b:.2f} = 0"
        ax.text(0.05, 0.95, equation, transform=ax.transAxes,
                fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        
        ax.set_title(title)
        ax.set_xlabel(self.x_var.get())
        ax.set_ylabel(self.y_var.get())
        ax.legend()
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = IrisClassifierApp(root)
    root.mainloop()