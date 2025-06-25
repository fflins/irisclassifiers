# ui/pages/page_mlp.py
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from logic import evaluators
import seaborn as sns
import matplotlib.pyplot as plt

from logic.classifier_controller import run_mlp_training

def show_confusion_matrix_popup(parent, expected, predictions):
    cm = confusion_matrix(expected, predictions, labels=list(sorted(set(expected))))
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Esperado")
    ax.set_title("Matriz de Confusão")
    plt.show()


class MLPPage(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        
        self.hidden_size = tk.IntVar(value=10)
        self.epochs = tk.IntVar(value=10000)
        self.learning_rate = tk.DoubleVar(value=0.1)
        
        control_frame = ttk.LabelFrame(self, text="Controles do MLP")
        control_frame.pack(fill='x', padx=5, pady=5)
        
        top_row_frame = ttk.Frame(control_frame)
        top_row_frame.pack(fill='x', pady=5)

        ttk.Label(top_row_frame, text="Neurônios Ocultos:").pack(side='left', padx=5)
        ttk.Entry(top_row_frame, textvariable=self.hidden_size, width=12).pack(side='left', padx=5)
        
        ttk.Label(top_row_frame, text="Épocas:").pack(side='left', padx=5)
        ttk.Entry(top_row_frame, textvariable=self.epochs, width=12).pack(side='left', padx=5)

        ttk.Label(top_row_frame, text="Taxa de Aprendizado:").pack(side='left', padx=5)
        ttk.Entry(top_row_frame, textvariable=self.learning_rate, width=12).pack(side='left', padx=5)

        ttk.Button(control_frame, text="Executar MLP", command=self.run_and_display).pack(pady=10)
        
        self.results_frame = ttk.Frame(self)
        self.results_frame.pack(fill='both', expand=True, pady=10)

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

        ttk.Button(info_frame, text="Matriz de Confusão",
                command=lambda: show_confusion_matrix_popup(self, expected, predictions)
                ).pack(pady=5)
    

    def clear_frame(self, frame):
        for widget in frame.winfo_children():
            widget.destroy()

    def run_and_display(self):
        self.clear_frame(self.results_frame)
        hidden_size = self.hidden_size.get()
        epochs = self.epochs.get()
        learning_rate = self.learning_rate.get()

        try:
            results = run_mlp_training(hidden_size, epochs, learning_rate)
            
            info_text = (
                f"Resultados do MLP:\n"
                f"Neurônios Ocultos: {hidden_size}\n"
                f"Épocas: {epochs}\n"
                f"Taxa de Aprendizado: {learning_rate}\n"
                f"Perda Final: {results['loss_history'][-1]:.6f}\n"
                f"Acurácia: {results['accuracy']:.4f}"
            )
            ttk.Label(self.results_frame, text=info_text, justify='left').pack(anchor='w', padx=10)
            
            fig = Figure(figsize=(6, 4), dpi=100)
            ax = fig.add_subplot(111)
            ax.plot(results['loss_history'], 'b-')
            ax.set_title("Perda por Época")
            ax.set_xlabel("Épocas")
            ax.set_ylabel("Perda (Entropia Cruzada)")
            ax.grid(True)
            
            canvas = FigureCanvasTkAgg(fig, master=self.results_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True, pady=10)

            # Salva os rótulos para avaliação
            self.expected = results['true_labels']
            self.predictions = results['predictions']

            # Botão de avaliação
            ttk.Button(self.results_frame, text="Mostrar Avaliação dos Classificadores",
                    command=lambda: self.show_evaluation_popup(self.expected, self.predictions)
                    ).pack(pady=5)


        except Exception as e:
            messagebox.showerror("Erro na Execução", str(e))