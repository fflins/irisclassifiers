# ui/pages/page_perceptron.py
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import precision_score, recall_score, f1_score

from logic.classifier_controller import run_perceptron_training


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

class PerceptronPage(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        
        self.class1 = tk.StringVar(value="setosa")
        self.class2 = tk.StringVar(value="versicolor")
        self.type = tk.StringVar(value="Normal")
        self.alpha = tk.DoubleVar(value=0.01) # Mudei para DoubleVar para Alpha (pode ser decimal)
        self.max_itr = tk.IntVar(value=150)
        
        control_frame = ttk.LabelFrame(self, text="Controles do Perceptron")
        control_frame.pack(fill='x', padx=5, pady=5)
        
        # Frame para a primeira linha de controles (Classes e Tipo)
        # Usamos 'grid' aqui para melhor alinhamento ou 'pack' com 'side=left' para cada grupo
        top_row_frame = ttk.Frame(control_frame)
        top_row_frame.pack(fill='x', pady=5)

        ttk.Label(top_row_frame, text="Classe 1:").pack(side='left', padx=5)
        ttk.Combobox(top_row_frame, textvariable=self.class1, values=["setosa", "versicolor", "virginica"], width=12).pack(side='left', padx=5)
        
        ttk.Label(top_row_frame, text="Classe 2:").pack(side='left', padx=5)
        ttk.Combobox(top_row_frame, textvariable=self.class2, values=["setosa", "versicolor", "virginica"], width=12).pack(side='left', padx=5)

        ttk.Label(top_row_frame, text="Tipo:").pack(side='left', padx=5)
        ttk.Combobox(top_row_frame, textvariable=self.type, values=["Normal","Delta"], width=12).pack(side='left', padx=5)

        # ---

        # Frame para a segunda linha de controles (Alpha e Máx. Iterações)
        bottom_row_frame = ttk.Frame(control_frame)
        bottom_row_frame.pack(fill='x', pady=5) # Este frame fará com que a linha comece abaixo

        ttk.Label(bottom_row_frame, text="Alpha:").pack(side='left', padx=5)
        ttk.Entry(bottom_row_frame, textvariable=self.alpha, width=12).pack(side='left', padx=5)

        ttk.Label(bottom_row_frame, text="Máx. Iterações:").pack(side='left', padx=5)
        ttk.Entry(bottom_row_frame, textvariable=self.max_itr, width=12).pack(side='left', padx=5)


        ttk.Button(control_frame, text="Executar perceptron", command=self.run_and_display).pack(pady=10) 
        
        self.results_frame = ttk.Frame(self)
        self.results_frame.pack(fill='both', expand=True, pady=10)
    
    def show_evaluation_popup(self, expected, predictions, multi_class):
        popup = tk.Toplevel(self)
        popup.title("Estatísticas de Avaliação")
        info_frame = ttk.Frame(popup)
        info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        import logic.evaluators as evaluators  # seu módulo personalizado

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

        # Botão para mostrar matriz de confusão (implemente essa função como sugeri)
        ttk.Button(info_frame, text="Matriz de Confusão",
           command=lambda: show_confusion_matrix_popup(self, expected, predictions)
          ).pack(pady=5)


    def clear_frame(self, frame):
        for widget in frame.winfo_children():
            widget.destroy()

    def run_and_display(self):
        self.clear_frame(self.results_frame)
        c1 = self.class1.get()
        c2 = self.class2.get()
        type = self.type.get()
        alpha = self.alpha.get()
        max_itr = self.max_itr.get()

        if c1 == c2:
            messagebox.showerror("Erro", "As duas classes devem ser diferentes.")
            return

        try:
            results = run_perceptron_training(c1, c2, type, alpha, max_itr)
            
    
            info_text = (
                f"Resultados para: {c1.capitalize()} vs {c2.capitalize()}\n"
                f"Erro Final: {results['errors'][-1]:.6f}\n"
                f"Acurácia: {results['accuracy']:.4f}\n"
                f"Pesos Finais: {[round(float(w), 3) for w in results['weights']]}"       
            )
            ttk.Label(self.results_frame, text=info_text, justify='left').pack(anchor='w', padx=10)
            
            # Plotar o gráfico de erro
            fig = Figure(figsize=(6, 4), dpi=100)
            ax = fig.add_subplot(111)
            ax.plot(results['errors'], 'r.-')
            ax.set_title("Erro por Época")
            ax.set_xlabel("Épocas")
            ax.set_ylabel("Erro Quadrático Médio")
            ax.grid(True)

            # Salvar rótulos reais e predições para avaliação
            self.expected = results["expected"]
            self.predictions = results["predictions"]

            # Determinar se é multi-classe (opcional, aqui é sempre binário)
            multi_class = False

            # Botão de avaliação
            ttk.Button(self.results_frame, text="Mostrar Avaliação",
                    command=lambda: self.show_evaluation_popup(self.expected, self.predictions, multi_class)
                    ).pack(pady=5)
            
            canvas = FigureCanvasTkAgg(fig, master=self.results_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True, pady=10)

        except Exception as e:
            messagebox.showerror("Erro na Execução", str(e))