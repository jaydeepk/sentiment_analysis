import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from typing import List, Tuple
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, model: torch.nn.Module, criterion: torch.nn.Module, optimizer: Optimizer, 
                 scheduler: _LRScheduler, train_loader: DataLoader, val_loader: DataLoader, patience: int = 5):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.patience = patience
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def run(self) -> None:
        best_val_loss = float('inf')
        epochs_without_improvement = 0

        for epoch in range(30):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate_epoch()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            self.scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pth')
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            if epochs_without_improvement >= self.patience:
                print("Early stopping triggered")
                break
        
        self.plot_training_history()

    def train_epoch(self) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        for batch in self.train_loader:
            self.optimizer.zero_grad()
            input_ids = batch['input_ids']
            labels = batch['label'].float()
            outputs = self.model(input_ids)
            loss = self.criterion(outputs.squeeze(), labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            predicted = torch.round(torch.sigmoid(outputs.squeeze()))
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        return total_loss / len(self.train_loader), 100 * correct / total

    def validate_epoch(self) -> Tuple[float, float]:
        return self.calculate_loss_and_accuracy(self.val_loader)

    def calculate_loss_and_accuracy(self, data_loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids']
                labels = batch['label'].float()
                outputs = self.model(input_ids)
                loss = self.criterion(outputs.squeeze(), labels)
                total_loss += loss.item()
                predicted = torch.round(torch.sigmoid(outputs.squeeze()))
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return total_loss / len(data_loader), 100 * correct / total

    def plot_training_history(self) -> None:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label='Train Accuracy')
        plt.plot(self.val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()