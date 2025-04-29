import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import data_loader
import bayes
import classifiers
import perceptron

class SampleClassifierForm:
    def __init__(self, root, parent_app):
        """Initialize the sample classifier form"""
        self.root = root
        self.parent_app = parent_app
        self.popup = tk.Toplevel(root)
        self.popup.title("Sample Classifier")
        self.popup.geometry("600x500")
        self.popup.resizable(True, True)
        
        # Load data for classification
        self.data, self.training_sample, self.test_sample, self.setosasData, self.versicolorData, \
            self.virginicaData, self.setosasMean, self.versicolorMean, self.virginicaMean = data_loader.load_data()
        
        # Calculate covariance matrices for Bayes classifier
        self.cov_setosas = bayes.calculate_covariance_matrix(self.setosasData, self.setosasMean)
        self.cov_versicolor = bayes.calculate_covariance_matrix(self.versicolorData, self.versicolorMean)
        self.cov_virginica = bayes.calculate_covariance_matrix(self.virginicaData, self.virginicaMean)
        
        # Initialize the UI
        self.create_widgets()
    
    def create_widgets(self):
        """Create all UI elements"""
        main_frame = ttk.Frame(self.popup, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Input section
        input_frame = ttk.LabelFrame(main_frame, text="Sample Features")
        input_frame.pack(fill=tk.X, padx=5, pady=10)
        
        # Feature input fields
        self.feature_vars = []
        features = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width']
        
        for i, feature in enumerate(features):
            feature_frame = ttk.Frame(input_frame)
            feature_frame.pack(fill=tk.X, padx=5, pady=5)
            
            ttk.Label(feature_frame, text=f"{feature}:", width=15).pack(side=tk.LEFT)
            var = tk.StringVar(value="")
            entry = ttk.Entry(feature_frame, textvariable=var, width=10)
            entry.pack(side=tk.LEFT, padx=5)
            
            self.feature_vars.append(var)
        
        # Classification buttons
        classify_frame = ttk.LabelFrame(main_frame, text="Classificador")
        classify_frame.pack(fill=tk.X, padx=5, pady=10)
        
        classify_button_frame = ttk.Frame(classify_frame)
        classify_button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(classify_button_frame, text="Classificar Amostra", 
                  command=self.classify_all).pack(fill=tk.X, padx=5, pady=5)
        
        # Results section
        results_frame = ttk.LabelFrame(main_frame, text="Resultado")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=10)
        
        self.results_text = tk.Text(results_frame, height=10, width=50, wrap=tk.WORD)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Close button
        ttk.Button(main_frame, text="Close", command=self.popup.destroy).pack(pady=10)

    
    def get_sample_values(self):
        try:
            # Convert input string values to float
            values = [float(var.get().replace(',', '.')) for var in self.feature_vars]
            
            # Check if all values are provided
            if any(np.isnan(values)) or len(values) != 4:
                messagebox.showerror("Erro", "Preencha os quatro campos.")
                return None
            
            return np.array(values)
        except ValueError:
            messagebox.showerror("Erro", "Insira valores numéricos validos.")
            return None
    
    def classify_with_perceptron(self, sample):
        # Train perceptron for each pair of classes
        results = {}
        class_pairs = [
            ("setosa", "versicolor"),
            ("setosa", "virginica"),
            ("versicolor", "virginica")
        ]
        
        votes = {"setosa": 0, "versicolor": 0, "virginica": 0}
        
        for class1, class2 in class_pairs:
            # Train the perceptron
            weights, _, _, _ = perceptron.perceptron(class1, class2, alpha=0.01, max_iterations=150)
            
            # Need to add a 1 for the bias term
            sample_with_bias = np.append(sample, 1)
            output = np.dot(weights, sample_with_bias)
            
            # Classify the sample
            predicted_class = class1 if output >= 0 else class2
            votes[predicted_class] += 1
            
            results[f"{class1} vs {class2}"] = predicted_class
        
        # Find the class with the most votes
        final_class = max(votes, key=votes.get)
        
        return final_class, results
    
    def classify_with_min_distance(self, sample):
        return classifiers.distancia_minima(sample, self.setosasMean, self.versicolorMean, self.virginicaMean)
    
    def classify_with_max_distance(self, sample):
        return classifiers.distancia_maxima(sample, self.setosasMean, self.versicolorMean, self.virginicaMean)
    
    def classify_with_bayes(self, sample):
        return bayes.predict_bayes(
            sample, 
            self.cov_setosas, 
            self.cov_versicolor, 
            self.cov_virginica, 
            self.setosasMean, 
            self.versicolorMean, 
            self.virginicaMean
        )
    
    def classify_all(self):
        sample = self.get_sample_values()
        if sample is None:
            return
        
        # Clear the results
        self.results_text.delete(1.0, tk.END)
        
        # Format sample values for display
        sample_str = ", ".join([f"{feat}: {val:.2f}" for feat, val in zip(
            ['Sepal length', 'Sepal width', 'Petal length', 'Petal width'], sample)])
        
        self.results_text.insert(tk.END, f"Amostra: [{sample_str}]\n\n")
        
        # Classify with minimum distance
        min_dist_result = self.classify_with_min_distance(sample)
        self.results_text.insert(tk.END, f"Distância Mínima: {min_dist_result}\n")
        
        # Classify with maximum distance
        max_dist_result = self.classify_with_max_distance(sample)
        self.results_text.insert(tk.END, f"Distância máxima: {max_dist_result}\n")
        
        # Classify with Bayes
        bayes_result = self.classify_with_bayes(sample)
        self.results_text.insert(tk.END, f"Bayes: {bayes_result}\n")
        
        # Classify with Perceptron
        perceptron_result, perceptron_details = self.classify_with_perceptron(sample)
        self.results_text.insert(tk.END, f"Perceptron (Maioria): {perceptron_result}\n")
        
        # Add perceptron details
        self.results_text.insert(tk.END, "\nPerceptron Details:\n")
        for pair, result in perceptron_details.items():
            self.results_text.insert(tk.END, f"  {pair}: {result}\n")
