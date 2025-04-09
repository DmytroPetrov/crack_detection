import torch


class EarlyStopping:
    def __init__(self, patience=5, verbose=True, delta=0.0001, path='checkpoint.pth'):
        """
        Args:
            patience (int): How many epochs to wait after last improvement.
            verbose (bool): to print changes ot not
            delta (float): Minimum change to qualify as an improvement.
            path (str): Path to save the best model.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            if self.verbose:
                print(f"Validation loss improved ({self.best_loss:.4f} --> {val_loss:.4f}). Saving model...")
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)

    def load_best(self, model):
        model.load_state_dict(torch.load(self.path))
