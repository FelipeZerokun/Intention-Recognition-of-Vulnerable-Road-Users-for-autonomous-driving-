import torch
from sklearn.metrics import accuracy_score, classification_report

def evaluate_model(model, data_loader, device):
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
        for inputs, labels in data_loader:
            # move inputs and labels to the same device as the model
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            # Get the predicted classes for each input
            _, preds = torch.max(outputs, 1)

            # Store the predictions and true labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate the accuracy
    accuracy = accuracy_score(all_labels, all_preds)

    # Get the classification report (precision, recall, F1-score)
    report = classification_report(all_labels, all_preds, target_names=['standing_still', 'walking'])

    # Return the evaluation metrics as a dictionary
    metrics = {
        'accuracy': accuracy,
        'classification_report': report
    }

    return metrics