from datasets import BicoherenceSequenceDataset
from model import (
    SINDySz,
    ConvSINDyEncoder,
    ConvSINDyDecoder,
)
from fullres_autoencoder import FullResAutoencoder

from torch.utils.data import DataLoader
import torch.utils.data as data
import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping


def build_conv_masked_ae(height: int, width: int, latent_dim: int):
    """Construct a masked convolutional encoder/decoder pair sharing one AE.

    Builds a single :class:`FullResAutoencoder` and passes it to both the
    :class:`ConvSINDyEncoder` and :class:`ConvSINDyDecoder` wrappers.  This
    ensures that every parameter participates in the computation graph:
    the encoder-half weights receive gradients from the SINDy path and the
    decoder-half weights receive gradients from the reconstruction path.
    Previously each wrapper instantiated its own independent
    :class:`FullResAutoencoder`, leaving the decoder-half of the encoder's AE
    and the encoder-half of the decoder's AE permanently dead (grad=None).
    """
    shared_ae = FullResAutoencoder(height=height, width=width, latent_dim=latent_dim)
    encoder = ConvSINDyEncoder(height=height, width=width, latent_dim=latent_dim, ae=shared_ae)
    decoder = ConvSINDyDecoder(height=height, width=width, latent_dim=latent_dim, ae=shared_ae)
    return encoder, decoder


def main(data_file, annotation_file, sample_rate=5000):

    # Sequence length (SINDy time axis T): number of consecutive same-epoch
    # bicoherence maps per sample.
    time_dim = 8 
    latent_features = 5
    poly_order = 2
    
    # CRITICAL: Time step between consecutive bicoherence maps in seconds.
    # Each map is computed over a 5-second window (epoch_size), but consecutive
    # maps in the annotation sequence are separated by this time step.
    # This is the dt used for finite-difference derivatives in the SINDy loss.
    map_time_step = 3.0  # seconds between consecutive bicoherence maps

    dataset = BicoherenceSequenceDataset(
        data_file=data_file,
        annotation_file=annotation_file,
        seq_len=time_dim,
        epoch_size=5.0,
        f_max=25.0,
        epoch_id_restriction=None,
        sample_rate=sample_rate,
    )

    # Grid size (H, W) is determined by the bicoherence computation; the conv
    # autoencoder and the SINDy `system_features` (= H*W) are sized to match.
    H, W = dataset.get_grid_size()
    system_features = H * W

    trv_set_size = int(len(dataset) * 0.8)

    trv_indices = list(range(trv_set_size))
    test_indices = list(range(trv_set_size, len(dataset)))

    trv_set = data.Subset(dataset, trv_indices)
    test_set = data.Subset(dataset, test_indices)

    # use 20% of training data for validation
    train_set_size = int(len(trv_set) * 0.8)
    valid_set_size = len(trv_set) - train_set_size

    # split the train set into two
    seed = torch.Generator().manual_seed(42)
    train_set, valid_set = data.random_split(
        trv_set, [train_set_size, valid_set_size], generator=seed
    )

    train_loader = DataLoader(train_set, batch_size=4)
    valid_loader = DataLoader(valid_set, batch_size=4)
    test_loader = DataLoader(test_set, batch_size=4)

    # Convolutional masked autoencoder condition.
    #
    # The conv encoder ingests masked H×W bicoherence maps and embeds each map
    # as a latent vector of dimension ``latent_features``; the stack of maps in
    # a sequence forms the SINDy time axis. The decoder reconstructs the masked
    # map from the latent. This replaces the shallow FAN-GRU encoder/decoder.
    conditions = [
        ("conv_masked_ae", build_conv_masked_ae),
    ]

    for name, build_ae in conditions:
        encoder, decoder = build_ae(H, W, latent_features)
        sindy_sz = SINDySz(
            time_dim=time_dim,
            system_features=system_features,
            latent_features=latent_features,
            poly_order=poly_order,
            encoder=encoder,
            decoder=decoder,
            lr=0.001,
            nan_check=True,
            use_dual_optimizers=True,
            # Pass the INTER-MAP time step (seconds between consecutive
            # bicoherence maps), NOT the raw EEG sample rate. The loss
            # functions convert this to dt = 1 / map_dt_hz for finite
            # differences. With map_time_step=3.0s, map_dt_hz=1/3 Hz.
            sample_rate=(1.0 / map_time_step),
            # Preserve the conv autoencoder's own initialization; the generic
            # equal-variance init is designed for Linear/GRU layers, not convs.
            reinit=False,
        ).to(torch.get_default_dtype())

        early_stopping = EarlyStopping(monitor="valid_loss", min_delta=0.001, patience=6, check_on_train_epoch_end=False)

        trainer = L.Trainer(
            max_epochs=100,
            log_every_n_steps=1,
            accelerator="gpu",
            devices=1,
            default_root_dir=f"/app/Repos/pytorchSINDySz/lightning_logs/{name}",
            callbacks=[early_stopping],
            fast_dev_run=False,
            logger=True,
        )
        trainer.fit(sindy_sz, train_loader, valid_loader)
    # trainer.test(sindy_sz, dataloaders=test_loader)


if __name__ == "__main__":
    main(
        "/app/Data/WR/WR5_Run4.hdf5",
        "/app/Data/WR/Annotations/260218_annotations_a.pkl",
    )
