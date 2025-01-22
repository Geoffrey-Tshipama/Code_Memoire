"""
L'Objectif premier est de parvenir à faire un choix en fonction de différents paramètres mise en avant
pour parvenir à faire un choix objectif entre le modèle best.pt jugé meilleur par le concepteur de yolov8
(Ultralytics) qui va enregistrer les poids ayant obtenu la meilleure fonction de perte pendant
l'entrainement et le modèle last.pt qui lui garde le dernier paramètre ou performance que le modèle a obtenu
à la dernière epoque durant l'entrainement.

Nous devons retenir que c'est sur l'ensemble de test de notre dataset que nos évaluons les performances de n
os modèles


"""




from ultralytics import YOLO
import os


def compare_models(best_model_path, last_model_path, test_data_path, img_size=640, device=0):
    """
    Compare the performance of two YOLO models: best and last.

    Parameters:
        best_model_path (str): Path to the best model file.
        last_model_path (str): Path to the last model file.
        test_data_path (str): Path to the test dataset file.
        img_size (int): Image size to use for inference.
        device (int): Device ID to use for inference (0 for the first GPU, 'cpu' for CPU).

    Returns:
        None
    """

    # Load models
    best_model = YOLO(best_model_path)
    last_model = YOLO(last_model_path)

    # Evaluate models
    print("Evaluating best model...")
    best_results = best_model.val(data=test_data_path, imgsz=img_size, device=device)

    print("Evaluating last model...")
    last_results = last_model.val(data=test_data_path, imgsz=img_size, device=device)

    # Extract metrics for comparison
    best_metrics = best_results  # Use correct attributes based on the latest Ultralytics library
    last_metrics = last_results  # Use correct attributes based on the latest Ultralytics library

    print("\nPerformance Comparison:")
    print("----------------------")
    print(f"Metric\t\t\tBest Model\tLast Model")
    print(f"mAP@0.5\t\t\t{best_metrics.box.map50:.4f}\t\t{last_metrics.box.map50:.4f}")
    # La précision avec un seuil de 50 sur toutes les classes confondues du modèle
    print(f"mAP@0.75\t\t\t{best_metrics.box.map75:.4f}\t\t{last_metrics.box.map75:.4f}")
    print(f"mAP@0.75\t\t\t{best_metrics.box.maps:.4f}\t\t{last_metrics.box.maps:.4f}")
    print(f"mAP@0.75\t\t\t{best_metrics.box.map:.4f}\t\t{last_metrics.box.map:.4f}")
    # print(f"Precision\t\t{best_metrics.box.p:.4f}\t\t{last_metrics.box.p:.4f}")
    # print(f"Recall\t\t\t{best_metrics.box.r:.4f}\t\t{last_metrics.box.r:.4f}")
    # print(f"F1-Score\t\t{best_metrics.box.f1:.4f}\t\t{last_metrics.box.f1:.4f}")
    # print(f"mAP@0.5:0.95\t\t{best_metrics.box.map:.4f}\t\t{last_metrics.box.map:.4f}")

    # Further comparisons or decision logic can be added here


# Example usage
if __name__ == '__main__':
    # Paths to model files and test data
    best_model_path = "best.pt"
    last_model_path = "last.pt"
    test_data_path = "data.yaml"

    # Compare models
    compare_models(best_model_path, last_model_path, test_data_path)
