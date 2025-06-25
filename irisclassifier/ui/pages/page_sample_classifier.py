import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import data_loader
from classificadores import lineares, bayes
from logic.classifier_controller import run_perceptron_training, run_mlp_training

class SampleClassifierPage(ttk.Frame):
    def __init__(self, parent, app_controller=None):
        super().__init__(parent)
        self.trained_artifacts = {}
        self.models_ready = False
        
        self.setup_ui()
        
    def setup_ui(self):
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill='both', expand=True)
        
        self.prepare_button = ttk.Button(main_frame, text="1. Preparar Todos os Classificadores", command=self.prepare_models)
        self.prepare_button.pack(pady=10)
        
        input_frame = ttk.LabelFrame(main_frame, text="2. Inserir Nova Amostra")
        input_frame.pack(fill='x', padx=5, pady=10)
        
        self.feature_entries = []
        features = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width']
        for feature in features:
            row = ttk.Frame(input_frame)
            row.pack(fill='x', padx=5, pady=2, anchor='center')
            ttk.Label(row, text=f"{feature}:", width=15).pack(side='left')
            entry = ttk.Entry(row, width=10)
            entry.pack(side='left')
            self.feature_entries.append(entry)

        self.classify_button = ttk.Button(main_frame, text="3. Classificar Amostra", command=self.classify_sample, state="disabled")
        self.classify_button.pack(pady=10)
        
        results_frame = ttk.LabelFrame(main_frame, text="Resultados da Classificação")
        results_frame.pack(fill='both', expand=True, padx=5, pady=10)
        
        self.results_text = tk.Text(results_frame, height=15, wrap=tk.WORD, font=("Courier New", 10))
        self.results_text.pack(fill='both', expand=True, padx=5, pady=5)
        self.results_text.config(state="disabled")

    def prepare_models(self):
        try:
            self.prepare_button.config(state="disabled", text="Preparando...")
            self.update_idletasks()

            _, _, _, setosasData, versicolorData, virginicaData, setosasMean, versicolorMean, virginicaMean = data_loader.load_data()
            self.trained_artifacts['linear_means'] = {"setosa": setosasMean, "versicolor": versicolorMean, "virginica": virginicaMean}
            
            setosa_cov = bayes.calculate_covariance_matrix(setosasData)
            versicolor_cov = bayes.calculate_covariance_matrix(versicolorData)
            virginica_cov = bayes.calculate_covariance_matrix(virginicaData)
            
            self.trained_artifacts['bayes'] = {
                'setosa_matrix': setosa_cov,
                'versicolor_matrix': versicolor_cov,
                'virginica_matrix': virginica_cov,
                'mean_setosas': setosasMean,
                'mean_versicolor': versicolorMean,
                'mean_virginica': virginicaMean
            }
            
            self.trained_artifacts['perceptron_normal'] = {}
            self.trained_artifacts['perceptron_delta'] = {}
            class_pairs = [("setosa", "versicolor"), ("setosa", "virginica"), ("versicolor", "virginica")]
            for c1, c2 in class_pairs:
                 res_normal = run_perceptron_training(c1, c2, rule_type='Normal', alpha=0.01, max_iter=150)
                 self.trained_artifacts['perceptron_normal'][f"{c1}_vs_{c2}"] = {'weights': res_normal['weights'], 'classes': (c1, c2)}
                 res_delta = run_perceptron_training(c1, c2, rule_type='Delta', alpha=0.01, max_iter=150)
                 self.trained_artifacts['perceptron_delta'][f"{c1}_vs_{c2}"] = {'weights': res_delta['weights'], 'classes': (c1, c2)}

            self.trained_artifacts['mlp'] = run_mlp_training()

            self.models_ready = True
            self.classify_button.config(state="normal")
            messagebox.showinfo("Sucesso", "Todos os classificadores clássicos estão prontos para uso.")
        except Exception as e:
            messagebox.showerror("Erro ao Preparar Modelos", str(e))
        finally:
            self.prepare_button.config(state="normal", text="1. Preparar Todos os Classificadores") 

    def classify_sample(self):
        if not self.models_ready:
            messagebox.showwarning("Atenção", "Por favor, prepare os classificadores primeiro.")
            return

        try:
            sample = np.array([float(e.get().replace(',', '.')) for e in self.feature_entries])
            if len(sample) != 4: raise ValueError("São necessários 4 valores.")
        except ValueError:
            messagebox.showerror("Erro de Entrada", "Por favor, insira 4 valores numéricos válidos.")
            return
            
        self.results_text.config(state="normal")
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Classificando a amostra: {sample}\n\n")

        means = self.trained_artifacts['linear_means']
        res_min = lineares.distancia_minima_geral(sample, means)
        res_max = lineares.distancia_maxima_geral(sample, means)
        self.results_text.insert(tk.END, f"{'Distância Mínima:':<30} {res_min.capitalize()}\n")
        self.results_text.insert(tk.END, f"{'Distância Máxima:':<30} {res_max.capitalize()}\n")

        bayes_params = self.trained_artifacts['bayes']
        res_bayes = bayes.predict_bayes(
            sample,
            bayes_params['setosa_matrix'],
            bayes_params['versicolor_matrix'],
            bayes_params['virginica_matrix'],
            bayes_params['mean_setosas'],
            bayes_params['mean_versicolor'],
            bayes_params['mean_virginica']
        )
        self.results_text.insert(tk.END, f"{'Bayesiano (QDA):':<30} {res_bayes.capitalize()}\n")

        votes_normal = {"setosa": 0, "versicolor": 0, "virginica": 0}
        sample_with_bias = np.append(sample, 1)
        for p_params in self.trained_artifacts['perceptron_normal'].values():
            pred_num = np.dot(p_params['weights'], sample_with_bias)
            c1, c2 = p_params['classes']
            pred_class = c1 if pred_num >= 0 else c2
            votes_normal[pred_class] += 1
        res_perceptron_normal = max(votes_normal, key=votes_normal.get)
        self.results_text.insert(tk.END, f"{'Perceptron Normal (Votação):':<30} {res_perceptron_normal.capitalize()}\n")
        
        votes_delta = {"setosa": 0, "versicolor": 0, "virginica": 0}
        for p_params in self.trained_artifacts['perceptron_delta'].values():
            pred_num = np.dot(p_params['weights'], sample_with_bias)
            c1, c2 = p_params['classes']
            pred_class = c1 if pred_num >= 0 else c2
            votes_delta[pred_class] += 1
        res_perceptron_delta = max(votes_delta, key=votes_delta.get)
        self.results_text.insert(tk.END, f"{'Perceptron Delta (Votação):':<30} {res_perceptron_delta.capitalize()}\n")

        if 'mlp' in self.trained_artifacts:
            mlp_artifacts = self.trained_artifacts['mlp']
            model = mlp_artifacts['model']
            scaler = mlp_artifacts['scaler']
            class_names = mlp_artifacts['class_names']
            
            sample_scaled = scaler.transform(sample.reshape(1, -1))
            
            pred_probs = model.forward(sample_scaled)
            pred_index = np.argmax(pred_probs)
            res_mlp = class_names[pred_index]
            self.results_text.insert(tk.END, f"{'MLP (Rede Neural):':<25} {res_mlp.capitalize()}\n")

        self.results_text.config(state="disabled")