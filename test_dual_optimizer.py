"""Smoke test for the dual-optimizer training mode in ``SINDySz``.

Runs a `fast_dev_run` Lightning fit using ``use_dual_optimizers=True`` and
verifies that the dual-optimizer plumbing (loss split, manual optimization,
and configure_optimizers list) all work together end-to-end.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
import lightning as L

from model import (
    SINDySz,
    ShallowFANGRUEncoder,
    ShallowFANGRUDecoder,
)


def main():
    # Dummy data: input_dim must be divisible by 10 for the encoder/decoder
    # bottleneck calculation (bottleneck_dim = input_dim // 10).
    num_samples = 64
    time_steps = 10
    input_dim = 50

    x_data = torch.randn(num_samples, time_steps, input_dim)
    dataset = TensorDataset(x_data)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

    encoder = ShallowFANGRUEncoder(input_dim=input_dim)
    decoder = ShallowFANGRUDecoder(output_dim=input_dim)

    model = SINDySz(
        time_dim=time_steps,
        system_features=input_dim,
        latent_features=encoder.bottleneck_dim,
        poly_order=2,
        encoder=encoder,
        decoder=decoder,
        use_dual_optimizers=True,
        sindy_lr=0.01,
        decoder_lr=0.02,
    )

    # Sanity checks before fit
    assert model.automatic_optimization is False, (
        "Manual optimization must be enabled in dual-optimizer mode"
    )
    assert hasattr(model, "sindy_criterion") and hasattr(model, "decoder_criterion"), (
        "Dual-mode criteria not set up"
    )
    assert not hasattr(model, "criterion"), (
        "Single-mode criterion should not exist in dual-optimizer mode"
    )

    trainer = L.Trainer(fast_dev_run=True, accelerator="cpu", devices=1)
    trainer.fit(model, train_dataloaders=train_loader)

    print("Dual optimizer test completed successfully!")


if __name__ == "__main__":
    main()
