import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_path):
    """
    Training the model for action recognition.val_size])
    Args:
        model (torch.nn.Module): The model to train.
        train_loader (torch.utils.data.DataLoader): The training data loader.
        val_loader (torch.utils.data.DataLoader): The validation data loader.
        criterion (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): Optimization algorithm.
        num_epochs (int): Number of epochs to train the model.
        device (torch.device): Device to run training on (CPU or GPU).
        save_path (str): Path to save the model.

    Returns:
        torch.nn.Module: The trained model.
    """
    best_val_loss = float("inf")
    train_loss_history = []
    val_loss_history = []
    scaler = GradScaler()

    model.to(device)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_train_loss = 0.0
        correct_train_preds = 0
        total_train_samples = 0

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)

        for inputs, labels in tqdm(train_loader, desc='Training', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_train_preds += (predicted == labels).sum().item()
            total_train_samples += labels.size(0)

        train_accuracy = correct_train_preds / total_train_samples
        epoch_train_loss = running_train_loss / len(train_loader)
        train_loss_history.append(epoch_train_loss)

        print(f"Training Loss: {epoch_train_loss:.4f} | Training Accuracy: {train_accuracy:.4f}")

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        correct_val_preds = 0
        total_val_samples = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc='Validation', leave=False):
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_val_preds += (predicted == labels).sum().item()
                total_val_samples += labels.size(0)

            val_accuracy = correct_val_preds / total_val_samples
            epoch_val_loss = running_val_loss / len(val_loader)
            val_loss_history.append(epoch_val_loss)

            print(f"Validation Loss: {epoch_val_loss:.4f} | Validation Accuracy: {val_accuracy:.4f}")

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                torch.save(model.state_dict(), save_path)
                print(f"Best model saved with validation loss: {best_val_loss: .4f} at {save_path}")

    print("\nTraining complete.")
    return model, train_loss_history, val_loss_history