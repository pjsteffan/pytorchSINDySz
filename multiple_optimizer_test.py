from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.utils.data as data
import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, Callback


class DummyDataset(data.Dataset):
    """A dummy dataset that yields (input, label) pairs."""

    def __init__(self, num_samples=128, input_size=10, classify_size=4):
        self.inputs = torch.randn(num_samples, input_size)
        # Random one-hot-ish targets for the classifier
        labels = torch.randint(0, classify_size, (num_samples,))
        self.labels = torch.nn.functional.one_hot(labels, num_classes=classify_size).float()

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


class MyModel(L.LightningModule):
    def __init__(self, input_size, hidden_size, classify_size):
        super().__init__()
        self.automatic_optimization = False  # ⚡ Crucial step
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)
        self.classifier = nn.Linear(hidden_size, classify_size)
        self.decoder_criterion = nn.functional.mse_loss
        self.classifier_criterion = nn.functional.mse_loss

    def configure_optimizers(self):
        opt_encdec = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=0.01,
        )
        opt_cls = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.classifier.parameters()),
            lr=0.02,
        )
        return [opt_encdec, opt_cls]

    def calculate_encoding_loss(self, batch):
        x, _ = batch
        out = self.encoder(x)
        out = self.decoder(out)
        loss = self.decoder_criterion(out, x)
        return loss

    def calculate_classifier_loss(self, batch):
        x, y = batch
        out = self.encoder(x)
        out = self.classifier(out)
        loss = self.classifier_criterion(out, y)
        return loss

    def training_step(self, batch, batch_idx):
        opt_encdec, opt_cls = self.optimizers()

        # --- Train Encoder/Decoder ---
        self.toggle_optimizer(opt_encdec)
        loss_encdec = self.calculate_encoding_loss(batch)
        self.manual_backward(loss_encdec)
        opt_encdec.step()
        opt_encdec.zero_grad()
        self.untoggle_optimizer(opt_encdec)

        # --- Train Classifier ---
        self.toggle_optimizer(opt_cls)
        loss_cls = self.calculate_classifier_loss(batch)
        self.manual_backward(loss_cls)
        opt_cls.step()
        opt_cls.zero_grad()
        self.untoggle_optimizer(opt_cls)

        self.log_dict({"loss_encdec": loss_encdec, "loss_cls": loss_cls}, prog_bar=True)


if __name__ == "__main__":
    input_size = 10
    hidden_size = 8
    classify_size = 4

    dataset = DummyDataset(num_samples=128, input_size=input_size, classify_size=classify_size)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = MyModel(input_size=input_size, hidden_size=hidden_size, classify_size=classify_size)

    trainer = L.Trainer(fast_dev_run=True, accelerator="gpu", devices=1)
    trainer.fit(model, train_dataloaders=train_loader)
