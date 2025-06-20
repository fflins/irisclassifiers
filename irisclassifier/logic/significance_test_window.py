# ui/popups/significance_test_window.py
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from .evaluator_controller import perform_significance_test

class SignificanceTestWindow:
    def __init__(self, parent):
        self.popup = tk.Toplevel(parent)
        self.popup.title("Teste de Significância (Z-Test)")
        
        main_frame = ttk.Frame(self.popup, padding=20)
        main_frame.pack(fill='both', expand=True)
        
        self.entries = {}
        for i in range(1, 3):
            ttk.Label(main_frame, text=f"Classificador {i}", font=("", 10, "bold")).grid(row=0, column=(i-1)*2, columnspan=2, pady=(0, 10))
            ttk.Label(main_frame, text="Coeficiente:").grid(row=1, column=(i-1)*2, sticky='w')
            self.entries[f'coef{i}'] = ttk.Entry(main_frame, width=12)
            self.entries[f'coef{i}'].grid(row=1, column=(i-1)*2 + 1, padx=5)
            
            ttk.Label(main_frame, text="Variância:").grid(row=2, column=(i-1)*2, sticky='w')
            self.entries[f'var{i}'] = ttk.Entry(main_frame, width=12)
            self.entries[f'var{i}'].grid(row=2, column=(i-1)*2 + 1, padx=5, pady=5)
        
        ttk.Button(main_frame, text="Calcular Significância", command=self.calculate).grid(row=3, column=0, columnspan=4, pady=20)

    def calculate(self):
        try:
            c1 = float(self.entries['coef1'].get())
            v1 = float(self.entries['var1'].get())
            c2 = float(self.entries['coef2'].get())
            v2 = float(self.entries['var2'].get())
            
            results = perform_significance_test(c1, v1, c2, v2)
            
            result_text = (f"Rejeita H0, há diferença significativa." if results['is_significant']
                           else "Não rejeita H0, não há diferença significativa.")
            
            messagebox.showinfo("Resultado do Teste Z", f"{result_text}\n\nValor Z = {results['z_score']:.4f}")

        except ValueError:
            messagebox.showerror("Erro de Entrada", "Por favor, insira valores numéricos válidos.")
        except Exception as e:
            messagebox.showerror("Erro de Cálculo", str(e))