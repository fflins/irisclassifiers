# ui/pages/page_data_viewer.py
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

class DataViewerPage(ttk.Frame):
    def __init__(self, parent, app_controller=None): # app_controller é opcional aqui
        super().__init__(parent)
        
        # Carrega os dados uma vez para obter os nomes das colunas e classes
        try:
            self.data = pd.read_csv("data.csv", decimal=",")
        except FileNotFoundError:
            messagebox.showerror("Erro", "Arquivo data.csv não encontrado no diretório raiz.")
            return

        self.features = list(self.data.columns[:-1]) # Todas as colunas exceto a última
        self.species = list(self.data['Species'].unique())

        self.species_vars = {name: tk.BooleanVar(value=True) for name in self.species}
        self.feature_x = tk.StringVar(value=self.features[0])
        self.feature_y = tk.StringVar(value=self.features[2])

        self.setup_widgets()
        self.update_plot()

    def setup_widgets(self):
        options_frame = ttk.LabelFrame(self, text="Opções de Visualização")
        options_frame.pack(fill='x', padx=10, pady=5)

        # Frame para selecionar espécies
        species_frame = ttk.Frame(options_frame)
        species_frame.pack(side='left', padx=10, pady=5)
        for name, var in self.species_vars.items():
            ttk.Checkbutton(species_frame, text=name.capitalize(), variable=var, 
                            command=self.update_plot).pack(anchor='w')

        # Frame para seleção de características
        feature_frame = ttk.Frame(options_frame)
        feature_frame.pack(side='left', padx=20, pady=5)
        ttk.Label(feature_frame, text="Eixo X:").grid(row=0, column=0, sticky='w')
        x_combo = ttk.Combobox(feature_frame, textvariable=self.feature_x, values=self.features, state="readonly")
        x_combo.grid(row=0, column=1, padx=5, pady=2)
        x_combo.bind("<<ComboboxSelected>>", lambda e: self.update_plot())

        ttk.Label(feature_frame, text="Eixo Y:").grid(row=1, column=0, sticky='w')
        y_combo = ttk.Combobox(feature_frame, textvariable=self.feature_y, values=self.features, state="readonly")
        y_combo.grid(row=1, column=1, padx=5, pady=2)
        y_combo.bind("<<ComboboxSelected>>", lambda e: self.update_plot())
        
        self.plot_frame = ttk.Frame(self)
        self.plot_frame.pack(fill='both', expand=True, padx=10, pady=10)

    def update_plot(self):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        selected_species = [name for name, var in self.species_vars.items() if var.get()]
        if not selected_species: return
        
        filtered_data = self.data[self.data['Species'].isin(selected_species)]
        
        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        colors = {'setosa': '#E41A1C', 'versicolor': '#377EB8', 'virginica': '#4DAF4A'}

        for species, group in filtered_data.groupby('Species'):
            ax.scatter(group[self.feature_x.get()], group[self.feature_y.get()],
                       c=colors[species], label=species, s=50, alpha=0.8, edgecolors='k')

        ax.set_xlabel(self.feature_x.get())
        ax.set_ylabel(self.feature_y.get())
        ax.set_title(f"{self.feature_x.get()} vs. {self.feature_y.get()}")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        toolbar = NavigationToolbar2Tk(canvas, self.plot_frame)
        toolbar.update()