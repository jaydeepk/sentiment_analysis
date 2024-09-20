import argparse
import torch
import torch.nn as nn
from training_data.imdb import IMDB
from models.simple_nn import SimpleNN
from trainer import Trainer

def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    imdb = IMDB(max_length=args.max_length, train_size=args.train_size)
    train_data, val_data, test_data = imdb.get_data()
    print(imdb.get_data_info())

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size)

    model = SimpleNN(imdb.get_vocab_size(), args.embedding_dim, args.hidden_dim, 1, dropout_rate=args.dropout_rate).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)

    trainer = Trainer(model, criterion, optimizer, scheduler, train_loader, val_loader, patience=args.patience)
    trainer.run()

    model.load_state_dict(torch.load('best_model.pth'))
    test_loss, test_accuracy = trainer.calculate_loss_and_accuracy(test_loader)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a simple neural network on IMDB dataset")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--train_size", type=int, default=20000, help="Number of training samples")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--embedding_dim", type=int, default=100, help="Embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--dropout_rate", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
    args = parser.parse_args()

    run(args)