import tkinter as tk
from tkinter import ttk
import perceptron
import matplotlib.pyplot as plt
import data_loader
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class PerceptronApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Perceptron para Classificação de Íris")
        self.root.geometry("800x600")
        
        # Control panel
        control_frame = ttk.LabelFrame(root, text="Controles")
        control_frame.pack(fill="x", expand=False, padx=10, pady=10)
        
        # Frames for results and graphs
        self.result_frame = ttk.LabelFrame(root, text="Resultados")
        self.result_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        # Buttons for different combinations
        btn1 = ttk.Button(control_frame, text="Setosa e Versicolor", 
                         command=lambda: self.run_perceptron("setosa", "versicolor"))
        btn1.grid(row=0, column=0, padx=5, pady=5)
        btn2 = ttk.Button(control_frame, text="Setosa e Virginica", 
                         command=lambda: self.run_perceptron("setosa", "virginica"))
        btn2.grid(row=0, column=1, padx=5, pady=5)
        btn3 = ttk.Button(control_frame, text="Virginica e Versicolor", 
                         command=lambda: self.run_perceptron("virginica", "versicolor"))
        btn3.grid(row=0, column=2, padx=5, pady=5)
        
        # Feature selection for visualization
        feature_names = ['Comprimento da Sépala', 'Largura da Sépala', 
                 'Comprimento da Pétala', 'Largura da Pétala']
        ttk.Label(control_frame, text="Característica X:").grid(row=1, column=0, padx=5, pady=5)
        self.feature_x_var = tk.StringVar(value=feature_names[0])  # Default value
        self.cb_feature_x = ttk.Combobox(control_frame, textvariable=self.feature_x_var, 
                                        values=feature_names, width=20)
        self.cb_feature_x.grid(row=1, column=1, padx=5, pady=5)
        ttk.Label(control_frame, text="Característica Y:").grid(row=1, column=2, padx=5, pady=5)
        self.feature_y_var = tk.StringVar(value=feature_names[1])  # Default value
        self.cb_feature_y = ttk.Combobox(control_frame, textvariable=self.feature_y_var, 
                                        values=feature_names, width=20)
        self.cb_feature_y.grid(row=1, column=3, padx=5, pady=5)
        
        # Area to display results
        self.result_text = tk.Text(self.result_frame, height=5, width=50)
        self.result_text.pack(side=tk.TOP, fill="x", expand=False, padx=10, pady=10)
        
        # Area for graphs
        self.figure, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 4))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.result_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill="both", expand=True, padx=10, pady=10)
    
    def run_perceptron(self, classe1, classe2):
        try:
            # Get selected feature indices
            feature_names = ['Comprimento da Sépala', 'Largura da Sépala',
                        'Comprimento da Pétala', 'Largura da Pétala']
            feature_x = feature_names.index(self.feature_x_var.get())
            feature_y = feature_names.index(self.feature_y_var.get())
            
            # Clear previous results
            self.result_text.delete(1.0, tk.END)
            self.ax1.clear()
            self.ax2.clear()
            feature_indices = [feature_x, feature_y]
            
            # Run perceptron
            weights, accuracy, history = perceptron.perceptron(classe1, classe2, feature_indices)
            
            # Display results in text area
            self.result_text.insert(tk.END, f"Classificação: {classe1} vs {classe2}\n")
            self.result_text.insert(tk.END, f"Pesos finais: {weights}\n")
            self.result_text.insert(tk.END, f"Acurácia: {accuracy:.4f}\n")
            
            self.ax1.plot(history)
            self.ax1.set_title('Acurácia por Iteração')
            self.ax1.set_xlabel('Iteração')
            self.ax1.set_ylabel('Acurácia')
            self.ax1.grid(True)
            

            values, classes = data_loader.join_classes(classe1, classe2)
            X = values[:, feature_indices]
            self.ax2.scatter(X[classes==0, 0], X[classes==0, 1], c='blue', label=classe1)
            self.ax2.scatter(X[classes==1, 0], X[classes==1, 1], c='red', label=classe2)
            
            if len(weights) >= 3: 
                w0, w1, bias = weights
                min_x, max_x = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
                
                if w1 != 0:  # Evitar divisão por zero
                    slope = -w0 / w1  # Coeficiente angular
                    intercept = -bias / w1  # Coeficiente linear
                    
                    # Gerar pontos para a linha
                    xs = np.array([min_x, max_x])
                    ys = slope * xs + intercept
                    
                    # Plotar a linha
                    self.ax2.plot(xs, ys, 'k-', label='Fronteira de Decisão')
                    
                    # Criar a equação da reta como texto
                    equation_text = f"x1 = {slope:.2f} * x0 + {intercept:.2f}"
                    
                    # Adicionar a equação como anotação no gráfico
                    self.ax2.text(
                        0.05, 0.95, equation_text, transform=self.ax2.transAxes,
                        fontsize=10, verticalalignment='top', 
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5)
                    )
            
            
            self.ax2.set_title(f'Dados: {classe1} vs {classe2}')
            self.ax2.set_xlabel(feature_names[feature_x])
            self.ax2.set_ylabel(feature_names[feature_y])
            self.ax2.legend()
            self.ax2.grid(True)
            
            # Update graphs
            self.figure.tight_layout()
            self.canvas.draw()
        except Exception as e:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Erro: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = PerceptronApp(root)
    root.mainloop()