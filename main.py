from datasets import WRsmallepoch
from model import (
    SINDySz,
    ShallowFANGRUAutoencoder,
)

from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.utils.data as data
import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping,Callback




def main(data_file, annotation_file, sample_rate=5000):

    dataset = WRsmallepoch(
        data_file=data_file,
        annotation_file=annotation_file,
        single_channel_flag=True,
        psd_flag=False,
        epoch_id_restriction=2,
        epoch_size=5.0,
        sample_rate=sample_rate,
    )

    trv_set_size = int(len(dataset) * 0.8)
    # test_set_size = len(dataset) - trv_set_size

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

    train_loader = DataLoader(train_set, batch_size=9)
    valid_loader = DataLoader(valid_set, batch_size=9)
    test_loader = DataLoader(test_set, batch_size=9)

    time_dim = 10
    system_features = 50
    latent_features = 5
    poly_order = 2
    batch_size = 2

    # GRU-based encoder/decoder condition.
    #
    # `ShallowFANGRUAutoencoder` replaces the MLP bottleneck/expander layers
    # with GRUs operating along the time dimension, while keeping the FAN
    # layers and Win/Wout projections over the feature dimension.
    #
    # Note: the autoencoder is defined over the *feature* dimension; the
    # bottleneck is `system_features // 10`, so `system_features` must be
    # large enough (>=10) to give a non-zero bottleneck.
    conditions = [
        ("shallow_fan_gru", ShallowFANGRUAutoencoder),
    ]

    for name, AE in conditions:
        ae = AE(system_features)
        sindy_sz = SINDySz(
            time_dim=time_dim,
            system_features=system_features,
            latent_features=latent_features,
            poly_order=poly_order,
            encoder=ae.encoder,
            decoder=ae.decoder,
            lr=0.00001,
            nan_check=True,
        ).to(torch.get_default_dtype())

        early_stopping = EarlyStopping(monitor="valid_loss", min_delta=0.001, patience=6, check_on_train_epoch_end=False)



        trainer = L.Trainer(
            max_epochs=100,
            log_every_n_steps=10,
            accelerator="gpu",
            devices=1,
            default_root_dir=f"/app/Repos/pytorchSINDySz/lightning_logs/{name}",
            callbacks=[early_stopping],
            fast_dev_run=False,
            gradient_clip_val=0.5,
            logger=True,
        )
        trainer.fit(sindy_sz, train_loader, valid_loader)
    # trainer.test(sindy_sz, dataloaders=test_loader)


if __name__ == "__main__":
    main(
        "/app/Data/WR/WR5_Run4.hdf5",
        "/app/Data/WR/Annotations/260218_annotations_a.pkl",
    )
