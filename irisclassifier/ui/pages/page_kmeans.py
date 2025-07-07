# ui/pages/page_kmeans.py
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.decomposition import PCA # Usaremos PCA para visualizar em 2D
from logic.classifier_controller import run_kmeans_clustering

class KMeansPage(ttk.Frame):
    def __init__(self, parent, app_controller=None):
        super().__init__(parent)
        
        self.k_value = tk.IntVar(value=3)
        self.max_iters = tk.IntVar(value=100)
        
        # --- Controles ---
        control_frame = ttk.LabelFrame(self, text="Controles do K-Means")
        control_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(control_frame, text="Número de Clusters (k):").pack(side='left', padx=(10, 5))
        ttk.Entry(control_frame, textvariable=self.k_value, width=5).pack(side='left')
        
        ttk.Label(control_frame, text="Máx. Iterações:").pack(side='left', padx=(20, 5))
        ttk.Entry(control_frame, textvariable=self.max_iters, width=8).pack(side='left')
        
        self.run_button = ttk.Button(control_frame, text="Executar K-Means", command=self.run_and_display)
        self.run_button.pack(side='left', padx=20)
        
        # --- Resultados ---
        self.results_frame = ttk.Frame(self)
        self.results_frame.pack(fill='both', expand=True, pady=10)

    def run_and_display(self):
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        try:
            self.run_button.config(state="disabled", text="Executando...")
            self.update_idletasks()

            k = self.k_value.get()
            max_iters = self.max_iters.get()
            results = run_kmeans_clustering(k, max_iters)
            
            # --- Exibir Resultados ---
            info_text = (
                f"K-Means executado com k={results['k']}.\n"
            )
            ttk.Label(self.results_frame, text=info_text, justify='left').pack(anchor='w', padx=10, pady=5)

            # --- Gráficos ---
            fig = Figure(figsize=(12, 6), dpi=100)
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)

            # Reduzir dados para 2D para visualização
            pca = PCA(n_components=2)
            data_2d = pca.fit_transform(results['scaled_data'])
            centroids_2d = pca.transform(results['centroids'])

            # Gráfico 1: Rótulos Verdadeiros
            scatter1 = ax1.scatter(data_2d[:, 0], data_2d[:, 1], c=results['true_labels'], cmap='viridis', alpha=0.8)
            ax1.set_title("Classes Reais (Verdade)")
            ax1.set_xlabel("Componente Principal 1")
            ax1.set_ylabel("Componente Principal 2")
            ax1.grid(True)
            ax1.legend(handles=scatter1.legend_elements()[0], labels=['Setosa', 'Versicolor', 'Virginica'])

            # Gráfico 2: Clusters Encontrados pelo K-Means
            scatter2 = ax2.scatter(data_2d[:, 0], data_2d[:, 1], c=results['cluster_labels'], cmap='viridis', alpha=0.8)
            # Plotar os centroides
            ax2.scatter(centroids_2d[:, 0], centroids_2d[:, 1], marker='X', s=200, c='red', edgecolor='black', label='Centroides')
            ax2.set_title(f"Clusters Encontrados (k={results['k']})")
            ax2.set_xlabel("Componente Principal 1")
            ax2.grid(True)
            ax2.legend()
            
            fig.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=self.results_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True, pady=10)

        except Exception as e:
            messagebox.showerror("Erro na Execução", str(e))
        finally:
            self.run_button.config(state="normal", text="Executar K-Means")