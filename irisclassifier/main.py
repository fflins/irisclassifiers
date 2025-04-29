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
from scipy import stats
import bayes
import utils
from sample_classifier import SampleClassifierForm

class PerceptronApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Classificador Perceptron")
        self.root.geometry("1200x700") 
        self.root.minsize(900, 600)
        
        # configurar  estilo
        style = ttk.Style()
        style.configure("TButton", padding=6, relief="flat", background="#ccc")
        style.configure("TFrame", background="#f0f0f0")
        style.configure("TLabelframe", background="#f0f0f0")
        
        # frame principal 
        main_frame = ttk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        control_frame = ttk.LabelFrame(main_frame, text="Controls")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # frame  perceptron
        perceptron_frame = ttk.LabelFrame(control_frame, text="Perceptron")
        perceptron_frame.pack(fill=tk.X, padx=5, pady=5)

        #frame parametros
        params_frame = ttk.LabelFrame(perceptron_frame, text="Parâmetros")
        params_frame.pack(fill=tk.X, padx=5, pady=10)
        
        #frame parametro alpha
        lr_frame = ttk.Frame(params_frame)
        lr_frame.pack(fill=tk.X, pady=5)
        ttk.Label(lr_frame, text="Alpha:").pack(side=tk.LEFT, padx=5)
        self.learning_rate = tk.StringVar(value="0.01")
        ttk.Entry(lr_frame, textvariable=self.learning_rate, width=8).pack(side=tk.RIGHT, padx=5)
        
        #frame parametro iteraçoes
        iter_frame = ttk.Frame(params_frame)
        iter_frame.pack(fill=tk.X, pady=5)
        ttk.Label(iter_frame, text="Máximo de iterações:").pack(side=tk.LEFT, padx=5)
        self.max_iterations = tk.StringVar(value="150")
        ttk.Entry(iter_frame, textvariable=self.max_iterations, width=8).pack(side=tk.RIGHT, padx=5)
        
        # botões perceptron 
        ttk.Button(perceptron_frame, text="Setosa vs Versicolor", 
                  command=lambda: self.show_results_perceptron("setosa", "versicolor")).pack(fill=tk.X, pady=5, padx=5)
        
        ttk.Button(perceptron_frame, text="Setosa vs Virginica", 
                  command=lambda: self.show_results_perceptron("setosa", "virginica")).pack(fill=tk.X, pady=5, padx=5)
        
        
        ttk.Button(perceptron_frame, text="Versicolor vs Virginica", 
                  command=lambda: self.show_results_perceptron("versicolor", "virginica")).pack(fill=tk.X, pady=5, padx=5)
        
        # frame dados
        data_viz_frame = ttk.LabelFrame(control_frame, text="Data")
        data_viz_frame.pack(fill=tk.X, padx=5, pady=10)
        
        ttk.Button(data_viz_frame, text="Exibir dados", 
                  command=self.show_data_window).pack(fill=tk.X, pady=5, padx=5)
        

        # frame classificadores lineares
        linear_classifiers_frame = ttk.LabelFrame(control_frame, text="Classificadores Lineares")
        linear_classifiers_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(linear_classifiers_frame, text="Classificador de Bayes", 
          command=lambda: self.show_results_bayes()).pack(fill=tk.X, pady=5, padx=5)

        ttk.Button(linear_classifiers_frame, text="Distância Mínima", 
                  command=lambda: self.show_results_linear("minimal")).pack(fill=tk.X, pady=5, padx=5)
        
        ttk.Button(linear_classifiers_frame, text="Distância Máxima", 
                  command=lambda: self.show_results_linear("maximal")).pack(fill=tk.X, pady=5, padx=5)
    


        #frame avaliaçao
        evaluation_frame = ttk.LabelFrame(control_frame, text="Avaliação")
        evaluation_frame.pack(fill=tk.X, padx=5, pady=5)


        #frame teste de significancia
        ttk.Button(evaluation_frame, text="Testar significância", 
                  command=lambda: self.evaluate_significance()).pack(fill=tk.X, pady=5, padx=5)

        sample_classifier_frame = ttk.LabelFrame(control_frame, text="Classificador")
        sample_classifier_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(sample_classifier_frame, text="Classificar nova amostra", 
          command=self.open_sample_classifier).pack(fill=tk.X, pady=5, padx=5)
        
        # frame resultados
        self.results_frame = ttk.LabelFrame(main_frame, text="Resultados")
        self.results_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # notebooks
        self.notebook = ttk.Notebook(self.results_frame)
        self.perceptron_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.perceptron_tab, text="Perceptron")
        self.data_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.data_tab, text="Classes")
        self.linear_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.linear_tab,text="Lineares")
        
        

        #funçao para executar o perceptron
    def show_results_perceptron(self, class1, class2):
        try:
            self.clear_frame(self.perceptron_tab)
            self.notebook.pack(fill=tk.BOTH, expand=True)
            self.notebook.select(0) 
            
            
            # obtem valores dos parâmetros
            alpha = float(self.learning_rate.get())
            max_iter = int(self.max_iterations.get())
            
            # treina o perceptron
            weights, errors, values_test, classes_test = perceptron.perceptron(
                class1, class2, alpha=alpha, max_iterations=max_iter)
            

            # previsões para avaliação
            predictions = [class1 if np.dot(weights, sample) >= 0 else class2 for sample in values_test]
            classes_test_str = [class1 if c == 1 else class2 for c in classes_test]

            # frame principal
            result_main = ttk.Frame(self.perceptron_tab)
            result_main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # frame para informações
            info_frame = ttk.LabelFrame(result_main, text="Informações de treino")
            info_frame.pack(fill=tk.X, pady=5)

            # botão "Avaliar"
            ttk.Button(info_frame, text="Avaliar", 
                  command=lambda: self.show_evaluation_popup(classes_test_str, predictions,multi_class=False)).pack(anchor="w", padx=10, pady=2)
            
            #implementar
           #ttk.Button(info_frame, text="Visualizar Superfície de Decisão", 
           #command=lambda: utils.visualize_perceptron_decision_surface(self, weights, class1, class2)).pack(anchor="w", padx=10, pady=2)
            
            # resultados do treinamento
            ttk.Label(info_frame, text=f"{class1.capitalize()} vs {class2.capitalize()}", 
                     font=("Arial", 10, "bold")).pack(anchor="w", padx=10, pady=5)
            ttk.Label(info_frame, text=f"Erro final: {errors[-1]:.6f}", 
                     font=("Arial", 10)).pack(anchor="w", padx=10, pady=2)

            
            # pesos finais
            weight_values = [round(float(w), 3) for w in weights]
            ttk.Label(info_frame, text=f"Vetor peso final: {weight_values}", 
                     font=("Arial", 10)).pack(anchor="w", padx=10, pady=2)
            
            # gráficos
            graphs_frame = ttk.Frame(result_main)
            graphs_frame.pack(fill=tk.BOTH, expand=True, pady=10)
            
            # frame para o gráfico de erro
            err_frame = ttk.LabelFrame(graphs_frame, text="Erro por Época: E(w) = (1/2)*(r-wᵀx)²")
            err_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
            
            # figura de erro por epoca
            fig_err = Figure(figsize=(6, 4))
            ax_err = fig_err.add_subplot(111)
            
            # plotar erro
            ax_err.plot(range(len(errors)), errors, marker='.', linestyle='-', markersize=3, color='red')
            ax_err.set_xlabel("Épocas")
            ax_err.set_ylabel("Erro")
            ax_err.set_title(f"Erro por Época ({class1} vs {class2})")
            ax_err.set_xlim(0, len(errors)-1)
            if len(errors) > 1:
                ax_err.set_ylim(0, max(errors) * 1.1)
            ax_err.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax_err.grid(True, linestyle='--', alpha=0.7)
            
            # adicionar canvas para erro
            canvas_err = FigureCanvasTkAgg(fig_err, master=err_frame)
            canvas_err.draw()
            canvas_err.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            toolbar_err = NavigationToolbar2Tk(canvas_err, err_frame)
            toolbar_err.update()
            
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    


    #funçao para exibir dados do dataset
    def show_data_window(self):
        self.clear_frame(self.data_tab)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        self.notebook.select(1)  
        
        #frame principal
        main_viz_frame = ttk.Frame(self.data_tab)
        main_viz_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        options_frame = ttk.LabelFrame(main_viz_frame, text="Opções")
        options_frame.pack(fill=tk.X, pady=5)
        
        #frame para selecionar especies a serem plotadas
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
        
        # frame para seleção de características
        feature_frame = ttk.LabelFrame(options_frame, text="Selecionar Características")
        feature_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.features = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width']
        
        self.feature_x = tk.StringVar(value=self.features[0])
        self.feature_y = tk.StringVar(value=self.features[2])
        
        # frame para eixo X
        x_frame = ttk.Frame(feature_frame)
        x_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(x_frame, text="X:").pack(side=tk.LEFT)
        x_combo = ttk.Combobox(x_frame, textvariable=self.feature_x, values=self.features, 
                              state="readonly", width=15)
        x_combo.pack(side=tk.RIGHT)
        x_combo.bind("<<ComboboxSelected>>", lambda e: self.update_data_plot())
        
        # frame para eixo Y
        y_frame = ttk.Frame(feature_frame)
        y_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(y_frame, text="Y:").pack(side=tk.LEFT)
        y_combo = ttk.Combobox(y_frame, textvariable=self.feature_y, values=self.features, 
                              state="readonly", width=15)
        y_combo.pack(side=tk.RIGHT)
        y_combo.bind("<<ComboboxSelected>>", lambda e: self.update_data_plot())
        
        # frame para o gráfico
        self.plot_frame = ttk.Frame(main_viz_frame)
        self.plot_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.update_data_plot()
    

    #funçao para atualizar o gráfico de dados
    def update_data_plot(self):
        self.clear_frame(self.plot_frame)
        
        try:
            #le dataset
            data = pd.read_csv("../data.csv", decimal=",")
            
            #usa as caracteristicas definidas
            x_feature = self.feature_x.get()
            y_feature = self.feature_y.get()
            
            selected_species = [species for species, var in self.species_vars.items() if var.get()]
            if not selected_species:
                ttk.Label(self.plot_frame, text="Please select at least one species!").pack(pady=20)
                return
            
            #filtra só as especies desejadas
            filtered_data = data[data['Species'].isin(selected_species)]
            

            #cria figura
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
    


    #funçao para os classificadores lineares
    def show_results_linear(self, classifier_type):
        self.clear_frame(self.linear_tab)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        self.notebook.select(2)  

        # obtem dados de teste
        data, training_sample, test_sample, setosasData, versicolorData, virginicaData, setosasMean, versicolorMean, virginicaMean = data_loader.load_data()
        test_data = test_sample.drop(columns="Species").values
        test_labels = test_sample["Species"].values

        # frame principal
        linear_main = ttk.Frame(self.linear_tab)
        linear_main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # frame para informações
        info_frame = ttk.LabelFrame(linear_main, text="Resultados Lineares")
        info_frame.pack(fill=tk.X, pady=5)

        if classifier_type == "minimal":
        # classificador de distância mínima
            predictions = [classifiers.distancia_minima(sample, setosasMean, versicolorMean, virginicaMean) for sample in test_data]
            cm = confusion_matrix(test_labels, predictions)

            # calcula a acurácia
            correct_predictions = sum(1 for true, pred in zip(test_labels, predictions) if true == pred)
            accuracy = correct_predictions / len(test_labels)

            ttk.Label(info_frame, text="Classificador de Distância Mínima:").pack(anchor="w", padx=10, pady=2)
            ttk.Label(info_frame, text=f"Acurácia: {accuracy:.4f}").pack(anchor="w", padx=10, pady=2) 

            


            ttk.Button(info_frame, text="Avaliar", 
                command=lambda: self.show_evaluation_popup(test_labels, predictions, multi_class = True)).pack(anchor="w", padx=10, pady=2)
            

        elif classifier_type == "maximal":
            # classificador de distância máxima
            predictions = [classifiers.distancia_maxima(sample, setosasMean, versicolorMean, virginicaMean) for sample in test_data]
            cm = confusion_matrix(test_labels, predictions)

            # calcular a acurácia 
            correct_predictions = sum(1 for true, pred in zip(test_labels, predictions) if true == pred)
            accuracy = correct_predictions / len(test_labels)

            ttk.Label(info_frame, text="Classificador de Distância Máxima:").pack(anchor="w", padx=10, pady=2)
            ttk.Label(info_frame, text=f"Acurácia: {accuracy:.4f}").pack(anchor="w", padx=10, pady=2)  # Exibir acurácia formatada
            ttk.Button(info_frame, text="Avaliar", 
                command=lambda: self.show_evaluation_popup(test_labels, predictions, multi_class=True)).pack(anchor="w", padx=10, pady=2)

    

    #funçao para exibir coeficientes
    def show_evaluation_popup(self, expected, predictions, multi_class):

        
        popup = tk.Toplevel(self.root)
        popup.title("Estatísticas de Avaliação")

        # frame popup para informações
        info_frame = ttk.Frame(popup)
        info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        if multi_class:
            kappa_value = evaluators.kappa(expected, predictions)
            kappa_var = evaluators.kappa_variance(expected, predictions)
            tau_value = evaluators.tau(expected, predictions)
            tau_var = evaluators.tau_variance(expected, predictions)
            
            ttk.Label(info_frame, text=f"Kappa: {kappa_value:.4f}").pack(anchor="w", padx=10, pady=2)
            ttk.Label(info_frame, text=f"Kappa Variance: {kappa_var:.4f}").pack(anchor="w", padx=10, pady=2)
            ttk.Label(info_frame, text=f"Tau: {tau_value:.4f}").pack(anchor="w", padx=10, pady=2)
            ttk.Label(info_frame, text=f"Tau Variance: {tau_var:.4f}").pack(anchor="w", padx=10, pady=2)
            
            ttk.Button(info_frame, text="Matriz de Confusão",
                    command=lambda: utils.show_confusion_matrix_popup(self, expected, predictions)).pack(pady=5)
        else:
        #calculos
            kappa_value = evaluators.kappa(expected, predictions)
            kappa_var = evaluators.kappa_variance(expected, predictions)
            tau_value = evaluators.tau(expected, predictions)
            tau_var = evaluators.tau_variance(expected, predictions)
            precision = evaluators.precision(expected, predictions)
            recall = evaluators.recall(expected, predictions)
            f1 = evaluators.f1_score(expected, predictions)  



            #exibir informações
            ttk.Label(info_frame, text=f"Kappa: {kappa_value:.4f}").pack(anchor="w", padx=10, pady=2)
            ttk.Label(info_frame, text=f"Kappa Variance: {kappa_var:.4f}").pack(anchor="w", padx=10, pady=2)
            ttk.Label(info_frame, text=f"Tau: {tau_value:.4f}").pack(anchor="w", padx=10, pady=2)
            ttk.Label(info_frame, text=f"Tau Variance: {tau_var:.4f}").pack(anchor="w", padx=10, pady=2)
            ttk.Label(info_frame, text=f"Precision: {precision:.2f}'").pack(anchor="w", padx=10, pady=2)
            ttk.Label(info_frame, text=f"Recall: '{recall:.2f}' ").pack(anchor="w", padx=10, pady=2)
            ttk.Label(info_frame, text=f"F1-score: '{f1:.2f}' ").pack(anchor="w", padx=10, pady=2)  

            
            ttk.Button(info_frame, text="Matriz de Confusão", 
                    command=lambda: utils.show_confusion_matrix_popup(self, expected, predictions)).pack(anchor="w", padx=10, pady=2)



    #testar significancia
    def evaluate_significance(self):
        popup = tk.Toplevel(self.root)
        popup.title("Teste de Significância")

        main_frame = ttk.Frame(popup)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # seleção do tipo de coeficiente (não que mude algo)
        ttk.Label(main_frame, text="Tipo de Coeficiente:").grid(row=0, column=0, columnspan=2, pady=5)
        coef_type = tk.StringVar(value="Kappa")
        coef_options = ttk.Combobox(main_frame, textvariable=coef_type, values=["Kappa", "Tau"], state="readonly")
        coef_options.grid(row=0, column=2, columnspan=2, pady=5)

        # entrada de dados 
        ttk.Label(main_frame, text="Classificador 1").grid(row=1, column=0, columnspan=2, pady=5)
        ttk.Label(main_frame, text="Valor:").grid(row=2, column=0, sticky="w")
        coef1_entry = ttk.Entry(main_frame, width=10)
        coef1_entry.grid(row=2, column=1, pady=2)
        ttk.Label(main_frame, text="Variância:").grid(row=3, column=0, sticky="w")
        var1_entry = ttk.Entry(main_frame, width=10)
        var1_entry.grid(row=3, column=1, pady=2)

        ttk.Label(main_frame, text="Classificador 2").grid(row=1, column=2, columnspan=2, pady=5)
        ttk.Label(main_frame, text="Valor:").grid(row=2, column=2, sticky="w")
        coef2_entry = ttk.Entry(main_frame, width=10)
        coef2_entry.grid(row=2, column=3, pady=2)
        ttk.Label(main_frame, text="Variância:").grid(row=3, column=2, sticky="w")
        var2_entry = ttk.Entry(main_frame, width=10)
        var2_entry.grid(row=3, column=3, pady=2)

        
        def calcular_e_exibir_significancia():
            try:
                coef1 = float(coef1_entry.get())
                var1 = float(var1_entry.get())
                coef2 = float(coef2_entry.get())
                var2 = float(var2_entry.get())
                coef_type_val = coef_type.get()

                z = (coef1 - coef2) / np.sqrt(var1 + var2)

                significativo = z > 1.96 or z < -1.96

                resultado_text = f"{coef_type_val}: {'Rejeita H0, há diferença significativa' if significativo else 'Não rejeita H0, não há diferença significativa'} (Z={z:.4f})"
                messagebox.showinfo("Resultados", resultado_text)

            except ValueError:
                messagebox.showerror("Erro", "Por favor, insira valores numéricos válidos.")
            except ZeroDivisionError:
                messagebox.showerror("Erro","Variâncias não podem ser 0")

        ttk.Button(main_frame, text="Calcular Significância", command=calcular_e_exibir_significancia).grid(row=4, column=0, columnspan=4, pady=10)

    #função para classificador de bayes
    def show_results_bayes(self):
        self.clear_frame(self.linear_tab)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        self.notebook.select(2)  

        # frame principal
        linear_main = ttk.Frame(self.linear_tab)
        linear_main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # frame para informações
        info_frame = ttk.LabelFrame(linear_main, text="Resultados:")
        info_frame.pack(fill=tk.X, pady=5)

        data, training_sample, test_sample, setosasData, versicolorData,virginicaData, setosasMean, versicolorMean, virginicaMean = data_loader.load_data()
        test_data = test_sample.drop(columns="Species").values
        test_labels = test_sample["Species"].values

        cov_setosas = bayes.calculate_covariance_matrix(setosasData, setosasMean)
        cov_versicolor = bayes.calculate_covariance_matrix(versicolorData, versicolorMean)
        cov_virginica = bayes.calculate_covariance_matrix(virginicaData, virginicaMean)


        # Classificador de Bayes Gaussiano
        predictions = [bayes.predict_bayes(sample, cov_setosas, cov_versicolor, cov_virginica, setosasMean, versicolorMean, virginicaMean) for sample in test_data]

        correct_predictions = sum(1 for true, pred in zip(test_labels, predictions) if true == pred)
        accuracy = correct_predictions / len(test_labels)

        ttk.Label(info_frame, text="Classificador de Bayes:").pack(anchor="w", padx=10, pady=2)
        ttk.Label(info_frame, text=f"Acurácia: {accuracy:.4f}").pack(anchor="w", padx=10, pady=2)  
        ttk.Button(info_frame, text="Avaliar", 
                command=lambda: self.show_evaluation_popup(test_labels, predictions, multi_class=True)).pack(anchor="w", padx=10, pady=2)

        decision_surface_frame = ttk.LabelFrame(linear_main, text="Equações de Superfície de Decisão:")
        decision_surface_frame.pack(fill=tk.X, pady=5)
    
    # Função para exibir a equação de superfície de decisão em uma popup
        def show_decision_surface(class1_name, class2_name, class1_cov, class2_cov, class1_mean, class2_mean):
            popup = tk.Toplevel(self.root)
            popup.title(f"Superfície de Decisão: {class1_name} vs {class2_name}")
            popup.geometry("800x400")
            
            # Frame para conteúdo da popup
            content_frame = ttk.Frame(popup, padding=10)
            content_frame.pack(fill=tk.BOTH, expand=True)
            
            # Título
            ttk.Label(content_frame, text=f"Superfície de Decisão entre {class1_name} e {class2_name}",
                    font=("Arial", 12, "bold")).pack(pady=10)
            
            # Obter a equação usando a função implementada
            try:
                equation = bayes.print_decision_surface_equation(
                    class1_cov, class2_cov, class1_mean, class2_mean)
                
                # Criar um widget de texto para mostrar a equação com formatação melhor
                text_widget = tk.Text(content_frame, wrap=tk.WORD, height=15, width=80)
                text_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
                

                text_widget.insert(tk.END, f"d({class1_name}) - d({class2_name}) = 0\n\n")
                text_widget.insert(tk.END, "Equação simplificada:\n")
                text_widget.insert(tk.END, f"{equation} = 0\n\n")
                
                
                # Configurar o widget como somente leitura
                text_widget.config(state=tk.DISABLED)
                
            except Exception as e:
                ttk.Label(content_frame, text=f"Erro ao calcular a equação: {str(e)}",
                        font=("Arial", 10)).pack(pady=10)
            
            # Botão de fechar
            ttk.Button(content_frame, text="Fechar", command=popup.destroy).pack(pady=10)
        
        # Adicionar botões para as três combinações possíveis
        ttk.Button(decision_surface_frame, text="Setosa vs Versicolor",
                command=lambda: show_decision_surface("setosa", "versicolor", cov_setosas, cov_versicolor, setosasMean, versicolorMean)
                ).pack(anchor="w", padx=10, pady=5)
        
        ttk.Button(decision_surface_frame, text="Setosa vs Virginica",
                command=lambda: show_decision_surface("setosa", "virginica", cov_setosas, cov_virginica, setosasMean, virginicaMean)
                ).pack(anchor="w", padx=10, pady=5)
        
        ttk.Button(decision_surface_frame, text="Versicolor vs Virginica",
                command=lambda: show_decision_surface("versicolor", "virginica", cov_versicolor, cov_virginica, versicolorMean, virginicaMean)
                ).pack(anchor="w", padx=10, pady=5)
        

    def open_sample_classifier(self):    
        SampleClassifierForm(self.root, self)


    


    def clear_frame(self, frame):
        for widget in frame.winfo_children():
            widget.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = PerceptronApp(root)
    root.mainloop()