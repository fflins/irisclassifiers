# ui/pages/page_perceptron.py
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from logic.classifier_controller import run_perceptron_training

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
            
            canvas = FigureCanvasTkAgg(fig, master=self.results_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True, pady=10)

        except Exception as e:
            messagebox.showerror("Erro na Execução", str(e))