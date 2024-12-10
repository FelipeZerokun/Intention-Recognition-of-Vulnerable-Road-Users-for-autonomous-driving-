import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import json
import pandas as pd

def evaluate_model(model, test_loader, device):
    """
    Evaluate the trained model on the test dataset.

    Args:
        model (torch.nn.Module): The trained model.
        data_loader (torch.utils.data.DataLoader): The test data loader.
        device (torch.device): Device to run evaluation on (CPU or GPU).

    Returns:
        metrics (dict): Dictionary with the evaluation metrics
    """
    # Put the model into evaluation mode
    model.eval()

    # Initialize variables for storing predictions and true labels
    all_preds = []
    all_labels = []

    # Disable gradient computation during evaluation
    with torch.no_grad():
        for inputs, labels in test_loader:
            # move inputs and labels to the same device as the model
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            # Get the predicted classes for each input
            _, preds = torch.max(outputs, 1)

            # Store the predictions and true labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Visualize Confusion matrix and save it
    visualize_results(all_preds, all_labels)
    # Get the results report and save it for future references
    save_report(all_preds, all_labels)


def visualize_results(y_pred, y_labels):
    """
    Visualize the results of the model predictions.

    Args:
        y_pred (list): List of predicted labels.
        y_labels (list): List of true labels.
    """
    # Confusion matrix
    cm = confusion_matrix(y_labels, y_pred)

    # Plot and save confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Standing", "Walking"],
                yticklabels=["Standing", "Walking"])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title("Confusion Matrix")
    plt.savefig("results/confusion_matrix.png")
    plt.close()

def save_report(y_pred, y_labels):
    """
    Save the predicted results and the ground truth labels as a CSV file.
    Then, calculate the accuracy and F1 score and save them in a JSON
    file.

    Args:
        y_pred (list): List of predicted labels.
        y_labels (list): List of true labels.
    """

    # Save predictions and true labels to CSV
    results_df = pd.DataFrame({"Ground Truth": y_labels, "Predictions": y_pred})
    results_df.to_csv("results/test_predictions.csv", index=False)

    # Generate classification report
    report = classification_report(y_labels, y_pred, target_names=["Standing", "Walking"], output_dict=True)

    # Save report to a JSON file
    with open("classification_report.json", "w") as f:
        json.dump(report, f, indent=4)

    # Log accuracy and other metrics
    accuracy = accuracy_score(y_labels, y_pred)

    with open("results/metrics_log.txt", "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Classification Report (Saved in classification_report.json)\n")