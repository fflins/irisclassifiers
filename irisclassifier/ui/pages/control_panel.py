# ui/control_panel.py
import tkinter as tk
from tkinter import ttk

class ControlPanel(ttk.Frame):
    def __init__(self, parent, app_controller):
        super().__init__(parent)
        self.app = app_controller

        perceptron_frame = ttk.LabelFrame(self, text="Perceptron")
        perceptron_frame.pack(fill=tk.X, padx=5, pady=5, anchor="n")
        
        ttk.Button(perceptron_frame, text="Executar Perceptron",
                   command=self.app.run_perceptron).pack(fill=tk.X, pady=5, padx=5)

        data_viz_frame = ttk.LabelFrame(self, text="Dados")
        data_viz_frame.pack(fill=tk.X, padx=5, pady=10, anchor="n")

        ttk.Button(data_viz_frame, text="Exibir Dados",
                   command=self.app.show_data_viewer).pack(fill=tk.X, pady=5, padx=5)

        linear_classifiers_frame = ttk.LabelFrame(self, text="Classificadores Lineares")
        linear_classifiers_frame.pack(fill=tk.X, padx=5, pady=5, anchor="n")
        
        ttk.Button(linear_classifiers_frame, text="Classificadores Lineares",
                   command=self.app.run_linear).pack(fill=tk.X, pady=5, padx=5)
        
        ttk.Button(linear_classifiers_frame, text="Classificador de Bayes",
                   command=self.app.run_bayes).pack(fill=tk.X, pady=5, padx=5)
        
        neural_frame = ttk.LabelFrame(self, text="Redes Neurais")
        neural_frame.pack(fill=tk.X, padx=5, pady=5, anchor="n")

        ttk.Button(neural_frame, text="Perceptron Multi-Camada",
                   command=self.app.run_mlp).pack(fill=tk.X, pady=5, padx=5)
        
        ttk.Button(neural_frame, text="Máquina de Boltzmann Restrita",
                   command=self.app.run_rbm).pack(fill=tk.X, pady=5, padx=5)

        evaluation_frame = ttk.LabelFrame(self, text="Avaliação")
        evaluation_frame.pack(fill=tk.X, padx=5, pady=5, anchor="n")

        ttk.Button(evaluation_frame, text="Testar Significância",
                   command=self.app.open_significance_test).pack(fill=tk.X, pady=5, padx=5)

        sample_classifier_frame = ttk.LabelFrame(self, text="Classificador")
        sample_classifier_frame.pack(fill=tk.X, padx=5, pady=5, anchor="n")

        ttk.Button(sample_classifier_frame, text="Classificar Nova Amostra",
                   command=self.app.show_sample_classifier).pack(fill=tk.X, pady=5, padx=5)
        