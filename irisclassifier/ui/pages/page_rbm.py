# ui/pages/page_rbm.py
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from logic.classifier_controller import run_rbm_training

class RBMPage(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        
        self.num_hidden = tk.IntVar(value=10)
        self.epochs = tk.IntVar(value=200)
        self.learning_rate = tk.DoubleVar(value=0.1)
        
        control_frame = ttk.LabelFrame(self, text="Controles da RBM")
        control_frame.pack(fill='x', padx=5, pady=5)
        
        top_row_frame = ttk.Frame(control_frame)
        top_row_frame.pack(fill='x', pady=5)

        ttk.Label(top_row_frame, text="Neurônios Ocultos:").pack(side='left', padx=5)
        ttk.Entry(top_row_frame, textvariable=self.num_hidden, width=12).pack(side='left', padx=5)
        
        ttk.Label(top_row_frame, text="Épocas:").pack(side='left', padx=5)
        ttk.Entry(top_row_frame, textvariable=self.epochs, width=12).pack(side='left', padx=5)

        ttk.Label(top_row_frame, text="Taxa de Aprendizado:").pack(side='left', padx=5)
        ttk.Entry(top_row_frame, textvariable=self.learning_rate, width=12).pack(side='left', padx=5)

        ttk.Button(control_frame, text="Executar RBM", command=self.run_and_display).pack(pady=10)
        
        self.results_frame = ttk.Frame(self)
        self.results_frame.pack(fill='both', expand=True, pady=10)

    def clear_frame(self, frame):
        for widget in frame.winfo_children():
            widget.destroy()

    def run_and_display(self):
        self.clear_frame(self.results_frame)
        num_hidden = self.num_hidden.get()
        epochs = self.epochs.get()
        learning_rate = self.learning_rate.get()

        try:
            results = run_rbm_training(num_hidden, epochs, learning_rate)
            
            info_text = (
                f"Resultados da RBM:\n"
                f"Neurônios Ocultos: {num_hidden}\n"
                f"Épocas: {epochs}\n"
                f"Taxa de Aprendizado: {learning_rate}\n"
                f"Erro de Reconstrução Final: {results['reconstruction_errors'][-1]:.6f}\n"
                f"Características Extraídas: {results['learned_features'].shape[1]} features"
            )
            ttk.Label(self.results_frame, text=info_text, justify='left').pack(anchor='w', padx=10)
            
            fig = Figure(figsize=(12, 4), dpi=100)
            
            ax1 = fig.add_subplot(121)
            ax1.plot(results['reconstruction_errors'], 'g-')
            ax1.set_title("Erro de Reconstrução por Época")
            ax1.set_xlabel("Épocas")
            ax1.set_ylabel("Erro Quadrático Médio")
            ax1.grid(True)
            
            ax2 = fig.add_subplot(122)
            colors = ['red', 'green', 'blue']
            class_names = ['Setosa', 'Versicolor', 'Virginica']
            
            for i, (color, name) in enumerate(zip(colors, class_names)):
                mask = results['original_labels'] == i
                if results['learned_features'].shape[1] >= 2:
                    ax2.scatter(results['learned_features'][mask, 0], 
                               results['learned_features'][mask, 1], 
                               c=color, label=name, alpha=0.7)
                    ax2.set_xlabel("Feature 1")
                    ax2.set_ylabel("Feature 2")
                else:
                    ax2.scatter(results['learned_features'][mask, 0], 
                               np.zeros_like(results['learned_features'][mask, 0]),
                               c=color, label=name, alpha=0.7)
                    ax2.set_xlabel("Feature 1")
                    ax2.set_ylabel("Constante")
            
            ax2.set_title("Características Aprendidas pela RBM")
            ax2.legend()
            ax2.grid(True)
            
            fig.tight_layout()
            
            canvas = FigureCanvasTkAgg(fig, master=self.results_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True, pady=10)

        except Exception as e:
            messagebox.showerror("Erro na Execução", str(e))