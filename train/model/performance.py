from dataclasses import dataclass


@dataclass
class EpochPerformance():
    epoch: int
    train_loss_primary: float
    train_accuracy_primary: float
    train_loss_auxiliary: float
    train_accuracy_auxiliary: float
    test_loss_primary: float
    test_accuracy_primary: float
    test_loss_auxiliary: float
    test_accuracy_auxiliary: float
    batch_id: int = 0

    def __str__(self):
        return f"Epoch {self.epoch}: " \
               f"Train Loss (Primary): {self.train_loss_primary:.4f}, " \
               f"Train Accuracy (Primary): {self.train_accuracy_primary:.4f}, " \
               f"Train Loss (Auxiliary): {self.train_loss_auxiliary:.4f}, " \
               f"Train Accuracy (Auxiliary): {self.train_accuracy_auxiliary:.4f}, " \
               f"Test Loss (Primary): {self.test_loss_primary:.4f}, " \
               f"Test Accuracy (Primary): {self.test_accuracy_primary:.4f}, " \
               f"Test Loss (Auxiliary): {self.test_loss_auxiliary:.4f}, " \
               f"Test Accuracy (Auxiliary): {self.test_accuracy_auxiliary:.4f}"\
               f"Batch ID: {self.batch_id}"