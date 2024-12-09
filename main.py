import torch
from torch.utils.data import DataLoader
from models.action_recognition_model import get_i3d_model
from training.ar_train import train_model
from utils import data_split
from training.ar_evaluate import evaluate_model

def main():
    # First, set the parameters
    num_epochs = 10
    batch_size = 8
    learning_rate = 0.001
    num_classes = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_dir = '/media/felipezero/T7 Shield/DATA/thesis/action_recognition_dataset'
    save_model_path = 'results/action_recognition_model.pth'

    # Get the datasets
    train_dataset, val_dataset, test_dataset = data_split.split_dataset(dataset_dir)

    # Data Loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load the model
    model = get_i3d_model(num_classes=num_classes)
    model.to(device)

    # Define the Loss and Optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                num_epochs=num_epochs,
                device=device,
                save_path=save_model_path
    )

    # Evaluate the model
    metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device
    )

    print("Evaluation metrics: ", metrics)


    print("Using device: ", device)


if __name__ == "__main__":
    main()