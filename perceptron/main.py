import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import perceptron
import pandas as pd
from matplotlib.ticker import MaxNLocator
import numpy as np

class PerceptronApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Classificador Perceptron")
        self.root.geometry("1200x700")  # Tamanho maior para acomodar gráficos
        self.root.minsize(900, 600)
        
        # Configurar o estilo
        style = ttk.Style()
        style.configure("TButton", padding=6, relief="flat", background="#ccc")
        style.configure("TFrame", background="#f0f0f0")
        style.configure("TLabelframe", background="#f0f0f0")
        
        # ===== Frame principal =====
        main_frame = ttk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        control_frame = ttk.LabelFrame(main_frame, text="Controls")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Frame para seleção do perceptron
        perceptron_frame = ttk.LabelFrame(control_frame, text="Perceptron")
        perceptron_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Botões do perceptron com descrições
        ttk.Button(perceptron_frame, text="Setosa vs Versicolor", 
                  command=lambda: self.show_results("setosa", "versicolor")).pack(fill=tk.X, pady=5, padx=5)
        
        ttk.Button(perceptron_frame, text="Setosa vs Virginica", 
                  command=lambda: self.show_results("setosa", "virginica")).pack(fill=tk.X, pady=5, padx=5)
        
        
        ttk.Button(perceptron_frame, text="Versicolor vs Virginica", 
                  command=lambda: self.show_results("versicolor", "virginica")).pack(fill=tk.X, pady=5, padx=5)
        
        # Frame para visualização de dados
        data_viz_frame = ttk.LabelFrame(control_frame, text="Data")
        data_viz_frame.pack(fill=tk.X, padx=5, pady=10)
        
        ttk.Button(data_viz_frame, text="Exibir dados", 
                  command=self.show_data_window).pack(fill=tk.X, pady=5, padx=5)
        
        
        #frame parametros
        params_frame = ttk.LabelFrame(control_frame, text="Parâmetros")
        params_frame.pack(fill=tk.X, padx=5, pady=10)
        
        #frame alpha
        lr_frame = ttk.Frame(params_frame)
        lr_frame.pack(fill=tk.X, pady=5)
        ttk.Label(lr_frame, text="Alpha:").pack(side=tk.LEFT, padx=5)
        self.learning_rate = tk.StringVar(value="0.01")
        ttk.Entry(lr_frame, textvariable=self.learning_rate, width=8).pack(side=tk.RIGHT, padx=5)
        
        #frame iteraçoes
        iter_frame = ttk.Frame(params_frame)
        iter_frame.pack(fill=tk.X, pady=5)
        ttk.Label(iter_frame, text="Máximo de iterações:").pack(side=tk.LEFT, padx=5)
        self.max_iterations = tk.StringVar(value="150")
        ttk.Entry(iter_frame, textvariable=self.max_iterations, width=8).pack(side=tk.RIGHT, padx=5)
        
        # ===== Frame direito para exibir resultados =====
        self.results_frame = ttk.LabelFrame(main_frame, text="Resultados")
        self.results_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Criar notebook para abas
        self.notebook = ttk.Notebook(self.results_frame)
        
        # Aba para resultados do perceptron
        self.perceptron_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.perceptron_tab, text="Perceptron")
        
        # Aba para visualização de dados
        self.data_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.data_tab, text="Classes")
        
        
    def show_results(self, class1, class2):
        try:
            self.clear_frame(self.perceptron_tab)
            self.notebook.pack(fill=tk.BOTH, expand=True)
            self.notebook.select(0) 
            
            
            # Obter valores dos parâmetros
            alpha = float(self.learning_rate.get())
            max_iter = int(self.max_iterations.get())
            
            # Treinar o perceptron
            weights, accuracies, values_test, classes_test = perceptron.perceptron(
                class1, class2, alpha=alpha, max_iterations=max_iter)
            
            # Frame principal
            result_main = ttk.Frame(self.perceptron_tab)
            result_main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Frame para informações
            info_frame = ttk.LabelFrame(result_main, text="Informações de treino")
            info_frame.pack(fill=tk.X, pady=5)
            
            # Resultados do treinamento
            ttk.Label(info_frame, text=f"{class1.capitalize()} vs {class2.capitalize()}", 
                     font=("Arial", 10, "bold")).pack(anchor="w", padx=10, pady=5)
            ttk.Label(info_frame, text=f"Acurácia: {accuracies[-1]:.3f}", 
                     font=("Arial", 10)).pack(anchor="w", padx=10, pady=2)
            
            # Exibir pesos finais
            weight_values = [round(float(w), 3) for w in weights]
            ttk.Label(info_frame, text=f"Vetor peso final: {weight_values}", 
                     font=("Arial", 10)).pack(anchor="w", padx=10, pady=2)
            
            # Frame para gráficos
            graphs_frame = ttk.Frame(result_main)
            graphs_frame.pack(fill=tk.BOTH, expand=True, pady=10)
            
            # Frame para o gráfico de acurácia
            acc_frame = ttk.LabelFrame(graphs_frame, text="Acurácia por Epócas")
            acc_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
            
            # Criar figura para acurácia
            fig_acc = Figure(figsize=(6, 4))
            ax_acc = fig_acc.add_subplot(111)
            
            # Plotar acurácia
            ax_acc.plot(range(len(accuracies)), accuracies, marker='.', linestyle='-', markersize=3)
            ax_acc.set_xlabel("Epochs")
            ax_acc.set_ylabel("Accuracy")
            ax_acc.set_title(f"Acurácia por Época ({class1} vs {class2})")
            ax_acc.set_xlim(0, len(accuracies)-1)
            ax_acc.set_ylim(min(accuracies)-0.05 if min(accuracies) < 1.0 else 0.95, 1.05)
            ax_acc.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax_acc.grid(True, linestyle='--', alpha=0.7)
            
            # Adicionar canvas para acurácia
            canvas_acc = FigureCanvasTkAgg(fig_acc, master=acc_frame)
            canvas_acc.draw()
            canvas_acc.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Barra de ferramentas para o gráfico
            toolbar_acc = NavigationToolbar2Tk(canvas_acc, acc_frame)
            toolbar_acc.update()
            
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
    def show_data_window(self):
        self.clear_frame(self.data_tab)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        self.notebook.select(1)  # Seleciona a aba de visualização
        
        main_viz_frame = ttk.Frame(self.data_tab)
        main_viz_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        options_frame = ttk.LabelFrame(main_viz_frame, text="Opções")
        options_frame.pack(fill=tk.X, pady=5)
        
        species_frame = ttk.LabelFrame(options_frame, text="Selecionar Especies")
        species_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=5, pady=5)
        
        self.species_vars = {
            "setosa": tk.BooleanVar(value=True),
            "versicolor": tk.BooleanVar(value=True),
            "virginica": tk.BooleanVar(value=True)
        }
        
        for species, var in self.species_vars.items():
            ttk.Checkbutton(species_frame, text=species.capitalize(), 
                           variable=var, command=self.update_data_plot).pack(anchor="w", padx=10, pady=2)
        
        # Frame para seleção de características
        feature_frame = ttk.LabelFrame(options_frame, text="Selecionar Características")
        feature_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Características disponíveis
        self.features = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width']
        
        # Variáveis para armazenar as características selecionadas
        self.feature_x = tk.StringVar(value=self.features[0])
        self.feature_y = tk.StringVar(value=self.features[2])
        
        # Frame para X
        x_frame = ttk.Frame(feature_frame)
        x_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(x_frame, text="X-axis:").pack(side=tk.LEFT)
        x_combo = ttk.Combobox(x_frame, textvariable=self.feature_x, values=self.features, 
                              state="readonly", width=15)
        x_combo.pack(side=tk.RIGHT)
        x_combo.bind("<<ComboboxSelected>>", lambda e: self.update_data_plot())
        
        # Frame para Y
        y_frame = ttk.Frame(feature_frame)
        y_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(y_frame, text="Y-axis:").pack(side=tk.LEFT)
        y_combo = ttk.Combobox(y_frame, textvariable=self.feature_y, values=self.features, 
                              state="readonly", width=15)
        y_combo.pack(side=tk.RIGHT)
        y_combo.bind("<<ComboboxSelected>>", lambda e: self.update_data_plot())
        
        # Frame para o gráfico
        self.plot_frame = ttk.Frame(main_viz_frame)
        self.plot_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.update_data_plot()
    
    def update_data_plot(self):
        """Atualiza o gráfico de visualização de dados"""
        self.clear_frame(self.plot_frame)
        
        try:
            data = pd.read_csv("../data.csv", decimal=",")
            
            x_feature = self.feature_x.get()
            y_feature = self.feature_y.get()
            
            selected_species = [species for species, var in self.species_vars.items() if var.get()]
            if not selected_species:
                ttk.Label(self.plot_frame, text="Please select at least one species!").pack(pady=20)
                return
            
            filtered_data = data[data['Species'].isin(selected_species)]
            
            fig = Figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
            
            colors = {'setosa': 'red', 'versicolor': 'blue', 'virginica': 'green'}
            markers = {'setosa': 'o', 'versicolor': 's', 'virginica': '^'}
            
            for species in selected_species:
                subset = filtered_data[filtered_data['Species'] == species]
                ax.scatter(subset[x_feature], subset[y_feature], 
                          c=colors[species], marker=markers[species], label=species, 
                          s=50, alpha=0.7, edgecolors='black')
            
            ax.set_xlabel(x_feature)
            ax.set_ylabel(y_feature)
            ax.set_title(f"{x_feature} vs {y_feature}")
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            
            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            toolbar = NavigationToolbar2Tk(canvas, self.plot_frame)
            toolbar.update()
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
    
    def clear_frame(self, frame):
        """Limpa todos os widgets de um frame"""
        for widget in frame.winfo_children():
            widget.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = PerceptronApp(root)
    root.mainloop()