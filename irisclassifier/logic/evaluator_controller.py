# logic/evaluation_controller.py

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score
from logic import evaluators 

def get_full_evaluation_report(y_true, y_pred, class_names=None):
    
    if class_names is None:
        class_names = [str(c) for c in sorted(list(set(y_true)))]

    # --- Métricas de Concordância ---
    
    # Usando a implementação robusta do Scikit-learn para Kappa
    kappa = cohen_kappa_score(y_true, y_pred)
    
    try:
        kappa_var = evaluators.kappa_variance(y_true, y_pred)
    except Exception:
        kappa_var = float('nan') # Retorna NaN se houver erro no cálculo

    tau = evaluators.tau(y_true, y_pred)
    tau_var = evaluators.tau_variance(y_true, y_pred)
    
    # --- Relatório de Classificação (Precisão, Revocação, F1-Score) ---

    report_str = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    report_dict = classification_report(y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True)
    
    # --- Matriz de Confusão ---

    labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    results = {
        "kappa": kappa,
        "kappa_variance": kappa_var,
        "tau": tau,
        "tau_variance": tau_var,
        "accuracy": report_dict["accuracy"],
        "classification_report_str": report_str,
        "classification_report_dict": report_dict,
        "confusion_matrix": cm,
        "labels_for_cm": labels 
    }
    
    return results

def perform_significance_test(coef1, var1, coef2, var2):
    """
    Calcula o teste Z de significância entre os resultados de dois classificadores.

    Args:
        coef1 (float): O coeficiente (Kappa ou Tau) do classificador 1.
        var1 (float): A variância do coeficiente do classificador 1.
        coef2 (float): O coeficiente (Kappa ou Tau) do classificador 2.
        var2 (float): A variância do coeficiente do classificador 2.

    Returns:
        dict: Um dicionário com o valor Z e um booleano indicando significância.
    """
    # A soma das variâncias no denominador não pode ser zero ou negativa
    if (var1 + var2) <= 0:
        raise ValueError("A soma das variâncias deve ser um número positivo.")
        
    z_score = (coef1 - coef2) / np.sqrt(var1 + var2)
    
    # Nível de significância de 5% (alfa=0.05). O valor crítico de Z para um teste bilateral é 1.96
    is_significant = abs(z_score) > 1.96
    
    return {
        "z_score": z_score,
        "is_significant": is_significant
    }