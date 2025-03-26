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
import data_loader
import classifiers
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import evaluators
import seaborn as sns

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

        #frame parametros
        params_frame = ttk.LabelFrame(perceptron_frame, text="Parâmetros")
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
        
        # Botões do perceptron com descrições
        ttk.Button(perceptron_frame, text="Setosa vs Versicolor", 
                  command=lambda: self.show_results_perceptron("setosa", "versicolor")).pack(fill=tk.X, pady=5, padx=5)
        
        ttk.Button(perceptron_frame, text="Setosa vs Virginica", 
                  command=lambda: self.show_results_perceptron("setosa", "virginica")).pack(fill=tk.X, pady=5, padx=5)
        
        
        ttk.Button(perceptron_frame, text="Versicolor vs Virginica", 
                  command=lambda: self.show_results_perceptron("versicolor", "virginica")).pack(fill=tk.X, pady=5, padx=5)
        
        # Frame para visualização de dados
        data_viz_frame = ttk.LabelFrame(control_frame, text="Data")
        data_viz_frame.pack(fill=tk.X, padx=5, pady=10)
        
        ttk.Button(data_viz_frame, text="Exibir dados", 
                  command=self.show_data_window).pack(fill=tk.X, pady=5, padx=5)
        

        #Frame lineares
        linear_classifiers_frame = ttk.LabelFrame(control_frame, text="Classificadores Lineares")
        linear_classifiers_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(linear_classifiers_frame, text="Distância Mínima", 
                  command=lambda: self.show_results_linear("minimal")).pack(fill=tk.X, pady=5, padx=5)
        
        ttk.Button(linear_classifiers_frame, text="Distância Máxima", 
                  command=lambda: self.show_results_linear("maximal")).pack(fill=tk.X, pady=5, padx=5)
    
        
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

        self.linear_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.linear_tab,text="Lineares")
        
        
    def show_results_perceptron(self, class1, class2):
        try:
            self.clear_frame(self.perceptron_tab)
            self.notebook.pack(fill=tk.BOTH, expand=True)
            self.notebook.select(0) 
            
            
            # Obter valores dos parâmetros
            alpha = float(self.learning_rate.get())
            max_iter = int(self.max_iterations.get())
            
            # Treinar o perceptron
            weights, errors, values_test, classes_test = perceptron.perceptron(
                class1, class2, alpha=alpha, max_iterations=max_iter)
            
            predictions = [class1 if np.dot(weights, sample) >= 0 else class2 for sample in values_test]

            # Converter classes_test para 1 e -1
            classes_test_str = [class1 if c == 1 else class2 for c in classes_test]

            # Frame principal
            result_main = ttk.Frame(self.perceptron_tab)
            result_main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Frame para informações
            info_frame = ttk.LabelFrame(result_main, text="Informações de treino")
            info_frame.pack(fill=tk.X, pady=5)

            # Botão "Avaliar"
            ttk.Button(info_frame, text="Avaliar", 
                  command=lambda: self.show_evaluation_popup(classes_test_str, predictions)).pack(anchor="w", padx=10, pady=2)
            # Resultados do treinamento
            ttk.Label(info_frame, text=f"{class1.capitalize()} vs {class2.capitalize()}", 
                     font=("Arial", 10, "bold")).pack(anchor="w", padx=10, pady=5)
            ttk.Label(info_frame, text=f"Erro final: {errors[-1]:.6f}", 
                     font=("Arial", 10)).pack(anchor="w", padx=10, pady=2)

            
            # Exibir pesos finais
            weight_values = [round(float(w), 3) for w in weights]
            ttk.Label(info_frame, text=f"Vetor peso final: {weight_values}", 
                     font=("Arial", 10)).pack(anchor="w", padx=10, pady=2)
            
            # Frame para gráficos
            graphs_frame = ttk.Frame(result_main)
            graphs_frame.pack(fill=tk.BOTH, expand=True, pady=10)
            
            # Frame para o gráfico de erro
            err_frame = ttk.LabelFrame(graphs_frame, text="Erro por Época: E(w) = (1/2)*(r-wᵀx)²")
            err_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
            
            # Criar figura para o erro
            fig_err = Figure(figsize=(6, 4))
            ax_err = fig_err.add_subplot(111)
            
            # Plotar erro
            ax_err.plot(range(len(errors)), errors, marker='.', linestyle='-', markersize=3, color='red')
            ax_err.set_xlabel("Épocas")
            ax_err.set_ylabel("Erro")
            ax_err.set_title(f"Erro por Época ({class1} vs {class2})")
            ax_err.set_xlim(0, len(errors)-1)
            if len(errors) > 1:
                ax_err.set_ylim(0, max(errors) * 1.1)
            ax_err.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax_err.grid(True, linestyle='--', alpha=0.7)
            
            # Adicionar canvas para erro
            canvas_err = FigureCanvasTkAgg(fig_err, master=err_frame)
            canvas_err.draw()
            canvas_err.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Barra de ferramentas para o gráfico
            toolbar_err = NavigationToolbar2Tk(canvas_err, err_frame)
            toolbar_err.update()
            
            
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
        ttk.Label(x_frame, text="X:").pack(side=tk.LEFT)
        x_combo = ttk.Combobox(x_frame, textvariable=self.feature_x, values=self.features, 
                              state="readonly", width=15)
        x_combo.pack(side=tk.RIGHT)
        x_combo.bind("<<ComboboxSelected>>", lambda e: self.update_data_plot())
        
        # Frame para Y
        y_frame = ttk.Frame(feature_frame)
        y_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(y_frame, text="Y:").pack(side=tk.LEFT)
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
    
    def show_results_linear(self, classifier_type):
        self.clear_frame(self.linear_tab)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        self.notebook.select(2)  # Seleciona a aba de lineares

        # Obter dados de teste
        data, training_sample, test_sample, setosasMean, versicolorMean, virginicaMean = data_loader.load_data()
        test_data = test_sample.drop(columns="Species").values
        test_labels = test_sample["Species"].values

        # Frame principal
        linear_main = ttk.Frame(self.linear_tab)
        linear_main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Frame para informações
        info_frame = ttk.LabelFrame(linear_main, text="Resultados Lineares")
        info_frame.pack(fill=tk.X, pady=5)

        if classifier_type == "minimal":
        # Classificador de distância mínima
            predictions = [classifiers.distancia_minima(sample, setosasMean, versicolorMean, virginicaMean) for sample in test_data]
            cm = confusion_matrix(test_labels, predictions)

            # Calcular a acurácia sem usar accuracy_score
            correct_predictions = sum(1 for true, pred in zip(test_labels, predictions) if true == pred)
            accuracy = correct_predictions / len(test_labels)

            ttk.Label(info_frame, text="Classificador de Distância Mínima:").pack(anchor="w", padx=10, pady=2)
            ttk.Label(info_frame, text=f"Acurácia: {accuracy:.4f}").pack(anchor="w", padx=10, pady=2)  # Exibir acurácia formatada
            ttk.Button(info_frame, text="Avaliar", 
                command=lambda: self.show_evaluation_popup(test_labels, predictions)).pack(anchor="w", padx=10, pady=2)
        elif classifier_type == "maximal":
            # Classificador de distância máxima
            predictions = [classifiers.distancia_maxima(sample, setosasMean, versicolorMean, virginicaMean) for sample in test_data]
            cm = confusion_matrix(test_labels, predictions)

            # Calcular a acurácia sem usar accuracy_score
            correct_predictions = sum(1 for true, pred in zip(test_labels, predictions) if true == pred)
            accuracy = correct_predictions / len(test_labels)

            ttk.Label(info_frame, text="Classificador de Distância Máxima:").pack(anchor="w", padx=10, pady=2)
            ttk.Label(info_frame, text=f"Acurácia: {accuracy:.4f}").pack(anchor="w", padx=10, pady=2)  # Exibir acurácia formatada
            ttk.Button(info_frame, text="Avaliar", 
                command=lambda: self.show_evaluation_popup(test_labels, predictions)).pack(anchor="w", padx=10, pady=2)

    
    def show_evaluation_popup(self, expected, predictions):
        popup = tk.Toplevel(self.root)
        popup.title("Estatísticas de Avaliação")

        # Frame para informações
        info_frame = ttk.Frame(popup)
        info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        kappa_value = evaluators.kappa(expected, predictions)
        kappa_var = evaluators.kappa_variance(expected, predictions)
        tau_value = evaluators.tau(expected, predictions)
        tau_var = evaluators.tau_variance(expected, predictions)
        precision_values = evaluators.precision(expected, predictions)
        recall_values = evaluators.recall(expected, predictions)

        ttk.Label(info_frame, text=f"Kappa: {kappa_value:.4f}").pack(anchor="w", padx=10, pady=2)
        ttk.Label(info_frame, text=f"Kappa Variance: {kappa_var:.4f}").pack(anchor="w", padx=10, pady=2)
        ttk.Label(info_frame, text=f"Tau: {tau_value:.4f}").pack(anchor="w", padx=10, pady=2)
        ttk.Label(info_frame, text=f"Tau Variance: {tau_var:.4f}").pack(anchor="w", padx=10, pady=2)
        ttk.Label(info_frame, text=f"Precision: {[f'{p:.2f}' for p in precision_values]}").pack(anchor="w", padx=10, pady=2)
        ttk.Label(info_frame, text=f"Recall: {[f'{r:.2f}' for r in recall_values]}").pack(anchor="w", padx=10, pady=2)

        # Botão para exibir matriz de confusão
        ttk.Button(info_frame, text="Matriz de Confusão", 
                command=lambda: self.show_confusion_matrix_popup(expected, predictions)).pack(anchor="w", padx=10, pady=2)


    def show_confusion_matrix_popup(self, expected, predictions):
        popup = tk.Toplevel(self.root)
        popup.title("Matriz de Confusão")

        # Frame para a matriz de confusão
        cm_frame = ttk.Frame(popup)
        cm_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        cm = confusion_matrix(expected, predictions)
        classes = np.unique(np.concatenate((expected, predictions)))

        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)

        # Transpor a matriz de confusão
        sns.heatmap(cm.T, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=classes, yticklabels=classes)

        # Inverter os rótulos dos eixos
        ax.set_xlabel('Valores Reais')
        ax.set_ylabel('Previsões')
        ax.set_title('Matriz de Confusão')

        canvas = FigureCanvasTkAgg(fig, master=cm_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def clear_frame(self, frame):
        for widget in frame.winfo_children():
            widget.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = PerceptronApp(root)
    root.mainloop()