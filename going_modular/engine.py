import numpy as np
import torch
from tqdm.auto import tqdm


def train(node_id, model, train_loader, val_loader, epochs, loss_fn, optimizer, scheduler=None, device="cpu",
          dp=False, delta=1e-5, max_physical_batch_size=256, privacy_engine=None, patience=2, save_model=None):

    # Create empty results dictionary
    results = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "stopping_n_epoch": None}
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in tqdm(range(epochs)):
        epoch_loss, epoch_acc = train_step(model, train_loader, loss_fn, optimizer, device, scheduler)
        tmp = ""

        val_loss, val_acc, _, _, _ = test(model, val_loader, loss_fn, device=device)

        # Print out what's happening
        print(
            f"\tNode: {node_id} \t"
            f"\tTrain Epoch: {epoch + 1} \t"
            f"Train_loss: {epoch_loss:.4f} | "
            f"Train_acc: {epoch_acc:.4f} % | "
            f"Validation_loss: {val_loss:.4f} | "
            f"Validation_acc: {val_acc:.4f} %" + tmp
        )

        # Update results dictionary
        results["train_loss"].append(epoch_loss)
        results["train_acc"].append(epoch_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save the best model weights
            torch.save(model.state_dict(), save_model)
            print(f"Model improved and saved to {save_model}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            results["stopping_n_epoch"] = epoch + 1
            break

    return results


def train_step(model, dataloader, loss_fn, optimizer, device, scheduler=None):
    # Put model in training mode
    model.train()
    accuracy = 0
    epoch_loss = 0
    total = 0
    correct = 0

    for x_batch, y_batch in dataloader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        # 1. Forward pass
        y_pred = model(x_batch)

        # 2. Calculate and accumulate loss
        loss = loss_fn(y_pred, y_batch)
        epoch_loss += loss.item()

        # Zero the gradients before running the backward pass.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 3. Calculate accuracy
        _, predicted = torch.max(y_pred, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

        # Print the current loss and accuracy
        if total > 0:
            accuracy = 100 * correct / total

    # Adjust metrics to get average loss and accuracy per batch
    epoch_loss = epoch_loss / len(dataloader)

    if scheduler:
        scheduler.step()
    else:
        print("\nNo scheduler")

    return epoch_loss, accuracy


def test(model, dataloader, loss_fn, device="cpu"):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    total = 0
    correct = 0
    y_pred = []
    y_true = []
    y_proba = []

    softmax = torch.nn.Softmax(dim=1)

    with torch.inference_mode():  # with torch.no_grad():  # Disable gradient computation
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            # 1. Forward pass
            output = model(x_batch)

            # 2. Calculate and accumulate probas
            probas_output = softmax(output)
            y_proba.extend(probas_output.detach().cpu().numpy())

            # 3. Calculate and accumulate loss
            loss = loss_fn(output, y_batch)
            total_loss += loss.item() * x_batch.size(0)

            # 4. Calculate and accumulate accuracy
            _, predicted = torch.max(output, 1)  # np.argmax(output.detach().cpu().numpy(), axis=1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            y_true.extend(y_batch.detach().cpu().numpy().tolist())  # Save Truth

            y_pred.extend(predicted.detach().cpu().numpy())  # Save Prediction

    model.train()
    # Calculate average loss and accuracy
    avg_loss = total_loss / total
    accuracy = 100 * correct / total

    return avg_loss, accuracy, y_pred, y_true, np.array(y_proba)
