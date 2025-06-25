# ui/results_notebook.py
from tkinter import ttk

class ResultsNotebook(ttk.Notebook):
    def __init__(self, parent):
        super().__init__(parent)
        self.pages = {} # Dicionário para guardar as páginas já criadas

    def add_page(self, PageClass, title):
        """ Adiciona uma nova página/aba ou seleciona uma existente. """
        if title in self.pages:
            self.select(self.pages[title])
        else:
            page_frame = ttk.Frame(self, padding="10")
            self.add(page_frame, text=title)
            
            # Instancia a classe da página específica dentro do frame da aba
            page_instance = PageClass(page_frame)
            page_instance.pack(fill='both', expand=True)

            self.pages[title] = page_frame
            self.select(page_frame)