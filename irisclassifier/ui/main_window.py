# ui/main_window.py
import tkinter as tk
from tkinter import ttk
from .pages.control_panel import ControlPanel
from .results_notebook import ResultsNotebook

class MainApplication:
    def __init__(self, root):
        self.root = root
        self.root.title("Classificador Iris Modularizado")
        self.root.geometry("1200x700")
        self.root.minsize(900, 600)
        self.trained_artifacts = {}

        style = ttk.Style()
        style.configure("TButton", padding=6, relief="flat", background="#ccc")
        style.configure("TFrame", background="#f0f0f0")
        style.configure("TLabelframe", background="#f0f0f0")

        self.learning_rate = tk.StringVar(value="0.01")
        self.max_iterations = tk.StringVar(value="150")

        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=3)

        control_panel = ControlPanel(main_frame, self)
        control_panel.grid(row=0, column=0, sticky="ns", padx=(0, 10))

        results_frame = ttk.LabelFrame(main_frame, text="Resultados")
        results_frame.grid(row=0, column=1, sticky="nsew")
        
        self.results_notebook = ResultsNotebook(results_frame)
        self.results_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def run_perceptron(self):
        from .pages.page_perceptron import PerceptronPage
        self.results_notebook.add_page(PerceptronPage, "Perceptron")
        
    def run_linear(self):
        from .pages.page_linear import LinearPage
        self.results_notebook.add_page(LinearPage, "Classificadores Lineares")
        
    def run_bayes(self):
        from .pages.page_bayes import BayesPage
        self.results_notebook.add_page(BayesPage, "Classificador Bayesiano")

    def run_mlp(self):
        from .pages.page_mlp import MLPPage
        self.results_notebook.add_page(MLPPage, "MLP (Rede Neural)")

    def run_rbm(self):
        from .pages.page_rbm import RBMPage
        self.results_notebook.add_page(RBMPage, "RBM (Análise de Features)")

    def show_data_viewer(self):
        from .pages.page_data_viewer import DataViewerPage
        self.results_notebook.add_page(DataViewerPage, "Visualizador de Dados")

    def open_significance_test(self):
        from logic.significance_test_window import SignificanceTestWindow
        SignificanceTestWindow(self.root) # Abre a popup

    def show_sample_classifier(self):
        from .pages.page_sample_classifier import SampleClassifierPage
        self.results_notebook.add_page(SampleClassifierPage, "Classificar Amostra")