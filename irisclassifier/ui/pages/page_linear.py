# ui/pages/page_linear.py
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score
from logic import evaluators  # seu módulo de kappa, tau, etc.

from logic.classifier_controller import run_linear_classifier

def show_confusion_matrix_popup(parent, expected, predictions):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(expected, predictions, labels=list(sorted(set(expected))))
    fig, ax = plt.subplots(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Esperado")
    ax.set_title("Matriz de Confusão")
    plt.show()


class LinearPage(ttk.Frame):
    def __init__(self, parent): # Removido app_controller por enquanto
        super().__init__(parent)
        
        self.classifier_type = tk.StringVar(value="minimal")
        
        # --- Widgets de Controle ---
        control_frame = ttk.LabelFrame(self, text="Controles dos Classificadores Lineares")
        control_frame.pack(fill='x', padx=5, pady=5)
        
        controls_row = ttk.Frame(control_frame)
        controls_row.pack(fill='x', pady=5, expand=True)
        
        ttk.Label(controls_row, text="Tipo de Classificador:").pack(side='left', padx=(10, 5))
        ttk.Combobox(controls_row, textvariable=self.classifier_type, 
                    values=["minimal", "maximal"], width=20, state="readonly").pack(side='left', padx=5)
        
        ttk.Button(controls_row, text="Executar Classificador", 
                  command=self.run_and_display).pack(side='left', padx=(20, 10))
        
        
        # --- Frame de Resultados com Notebook ---
        self.results_frame = ttk.Frame(self)
        self.results_frame.pack(fill='both', expand=True, pady=10)
        
        self.notebook = ttk.Notebook(self.results_frame)
        # O notebook será populado após a execução

    def clear_notebook(self):
        for tab in self.notebook.tabs():
            self.notebook.forget(tab)
        self.notebook.pack_forget()

    def show_evaluation_popup(self, expected, predictions):
        popup = tk.Toplevel(self)
        popup.title("Estatísticas de Avaliação")
        info_frame = ttk.Frame(popup)
        info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        kappa_value = evaluators.kappa(expected, predictions)
        kappa_var = evaluators.kappa_variance(expected, predictions)
        tau_value = evaluators.tau(expected, predictions)
        tau_var = evaluators.tau_variance(expected, predictions)

        ttk.Label(info_frame, text=f"Kappa: {kappa_value:.4f}").pack(anchor="w", pady=2)
        ttk.Label(info_frame, text=f"Kappa Variance: {kappa_var:.4f}").pack(anchor="w", pady=2)
        ttk.Label(info_frame, text=f"Tau: {tau_value:.4f}").pack(anchor="w", pady=2)
        ttk.Label(info_frame, text=f"Tau Variance: {tau_var:.4f}").pack(anchor="w", pady=2)

        if len(set(expected)) == 2:
            precision = precision_score(expected, predictions, pos_label=expected[0])
            recall = recall_score(expected, predictions, pos_label=expected[0])
            f1 = f1_score(expected, predictions, pos_label=expected[0])

            ttk.Label(info_frame, text=f"Precision: {precision:.2f}").pack(anchor="w", pady=2)
            ttk.Label(info_frame, text=f"Recall: {recall:.2f}").pack(anchor="w", pady=2)
            ttk.Label(info_frame, text=f"F1-score: {f1:.2f}").pack(anchor="w", pady=2)

        # Botão para mostrar a matriz de confusão em janela separada
        ttk.Button(info_frame, text="Matriz de Confusão",
                command=lambda: show_confusion_matrix_popup(self, expected, predictions)
                ).pack(pady=5)


    def run_and_display(self):
        self.clear_notebook()
        classifier_type = self.classifier_type.get()
        
        try:
            results = run_linear_classifier(classifier_type)
            self.notebook.pack(fill='both', expand=True) # Mostra o notebook
            
            # --- Aba 1: Resultados e Matriz de Confusão ---
            results_tab = ttk.Frame(self.notebook)
            self.notebook.add(results_tab, text="Resultados Gerais")
            
            type_name = "Distância Mínima" if classifier_type == "minimal" else "Distância Máxima"
            
            info_text = (
                f"Classificador: {type_name}\n"
                f"Acurácia: {results['accuracy']:.4f}\n"
                f"Total de Amostras de Teste: {len(results['predictions'])}"
            )
            ttk.Label(results_tab, text=info_text, justify='left').pack(anchor='w', padx=10, pady=10)
            
            class_names = results['class_names']
            cm = confusion_matrix(results['true_labels'], results['predictions'], labels=class_names)
            self.expected = results['true_labels']
            self.predictions = results['predictions']
            
            ttk.Button(results_tab, text="Mostrar Avaliação dos Classificadores",
                    command=lambda: self.show_evaluation_popup(self.expected, self.predictions)
                    ).pack()



            # --- Aba 2: Superfícies de Decisão ---
            surfaces_tab = ttk.Frame(self.notebook)
            self.notebook.add(surfaces_tab, text="Superfícies de Decisão")
            
            surfaces_text = scrolledtext.ScrolledText(surfaces_tab, wrap=tk.WORD, height=20, font=("Courier New", 10))
            surfaces_text.pack(fill='both', expand=True, padx=10, pady=10)
            
            surfaces_content = "Equações das Superfícies de Decisão Lineares (d(c1) - d(c2) = 0):\n\n"
            for pair, equation in results['decision_surfaces'].items():
                surfaces_content += f"--- {pair} ---\n{equation} = 0\n\n"
            
            surfaces_text.insert(tk.END, surfaces_content)
            surfaces_text.config(state=tk.DISABLED)

            


        except Exception as e:
            messagebox.showerror("Erro na Execução", str(e))