import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import config


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(64, 32, 16), n_classes=2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_mlp(X_train, y_train, X_val, y_val, max_epochs=None, lr=1e-3, epoch_callback=None):
    """Full-batch training loop. After every epoch, computes train/val
    accuracy plus per-example val predictions and softmax confidence,
    and hands them to epoch_callback(epoch, train_accuracy, val_accuracy,
    val_predictions, val_confidences, model) for snapshotting."""
    max_epochs = max_epochs or config.MAX_EPOCHS

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    n_classes = len(np.unique(y_train))
    model = MLP(input_dim=X_train.shape[1], n_classes=n_classes)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train_t)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            train_preds = model(X_train_t).argmax(dim=1)
            train_accuracy = (train_preds == y_train_t).float().mean().item()

            val_logits = model(X_val_t)
            val_probs = torch.softmax(val_logits, dim=1)
            val_preds = val_logits.argmax(dim=1)
            val_confidences = val_probs.gather(1, val_preds.unsqueeze(1)).squeeze(1)
            val_accuracy = (val_preds == y_val_t).float().mean().item()

        if epoch_callback is not None:
            epoch_callback(
                epoch=epoch,
                train_accuracy=train_accuracy,
                val_accuracy=val_accuracy,
                val_predictions=val_preds.numpy(),
                val_confidences=val_confidences.numpy(),
                model=model,
            )

    return model


def predict(model, X):
    """Predicted class ids for arbitrary points in original feature
    space (used for decision-boundary grid evaluation)."""
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32)
        return model(X_t).argmax(dim=1).numpy()
