# main.py
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import torch
import torch.utils.data as data
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from torch.utils.data import DataLoader
import multiprocessing as mp

from datasets import BicoherenceSequenceDataset
from model import SINDySz, ConvSINDyEncoder, ConvSINDyDecoder
from fullres_autoencoder import FullResAutoencoder


N_GPUS = 4
DATA_FILE = "/app/Data/WR/WR5_Run4.hdf5"
ANNOTATION_FILE = "/app/Data/WR/Annotations/260218_annotations_a.pkl"
SAMPLE_RATE = 5000
LOG_ROOT = "/app/Repos/pytorchSINDySz/lightning_logs/optuna"
DB_PATH = "sqlite:///optuna_sindy.db"


def build_conv_masked_ae(height: int, width: int, latent_dim: int):
    shared_ae = FullResAutoencoder(height=height, width=width, latent_dim=latent_dim)
    encoder = ConvSINDyEncoder(height=height, width=width, latent_dim=latent_dim, ae=shared_ae)
    decoder = ConvSINDyDecoder(height=height, width=width, latent_dim=latent_dim, ae=shared_ae)
    return encoder, decoder


def make_dataloaders(time_dim, map_time_step, batch_size):
    dataset = BicoherenceSequenceDataset(
        data_file=DATA_FILE,
        annotation_file=ANNOTATION_FILE,
        seq_len=time_dim,
        epoch_size=map_time_step,
        segment_seconds=1,
        f_max=25.0,
        epoch_id_restriction=None,
        sample_rate=SAMPLE_RATE,
    )
    H, W = dataset.get_grid_size()

    trv_size = int(len(dataset) * 0.8)
    trv_set = data.Subset(dataset, list(range(trv_size)))
    test_set = data.Subset(dataset, list(range(trv_size, len(dataset))))

    train_size = int(len(trv_set) * 0.8)
    valid_size = len(trv_set) - train_size
    seed = torch.Generator().manual_seed(42)
    train_set, valid_set = data.random_split(trv_set, [train_size, valid_size], generator=seed)

    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=2, persistent_workers=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=2, persistent_workers=True)
    return train_loader, valid_loader, H, W, dataset


def objective(trial, gpu_queue):
    gpu_id = gpu_queue.get()
    try:
        # ── Hyperparameters to search ─────────────────────────────────────────
        lr            = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        sindy_lr      = trial.suggest_float("sindy_lr", 1e-5, 1e-2, log=True)
        decoder_lr    = trial.suggest_float("decoder_lr", 1e-5, 1e-2, log=True)
        latent_features = trial.suggest_categorical("latent_features", [6, 9, 12, 16])
        poly_order    = trial.suggest_int("poly_order", 1, 3)
        time_dim      = trial.suggest_categorical("time_dim", [8, 16, 32])
        batch_size    = trial.suggest_categorical("batch_size", [2, 4, 8])
        map_time_step = trial.suggest_categorical("map_time_step", [1.0, 3.0, 5.0])

        train_loader, valid_loader, H, W, _ = make_dataloaders(time_dim, map_time_step, batch_size)
        system_features = H * W

        encoder, decoder = build_conv_masked_ae(H, W, latent_features)
        sindy_sz = SINDySz(
            time_dim=time_dim,
            system_features=system_features,
            latent_features=latent_features,
            poly_order=poly_order,
            encoder=encoder,
            decoder=decoder,
            lr=lr,
            sindy_lr=sindy_lr,
            decoder_lr=decoder_lr,
            nan_check=False,          # keep fast during search
            use_dual_optimizers=True,
            sample_rate=(1.0 / map_time_step),
            reinit=False,
        ).to(torch.get_default_dtype())

        # ── Callbacks ─────────────────────────────────────────────────────────
        # PyTorchLightningPruningCallback only works with automatic optimization.
        # With use_dual_optimizers=True (manual optimization), Lightning does NOT
        # call optimizer_step, so the standard pruning callback's check_pruned()
        # will raise TrialPruned at the wrong time.
        #
        # Safe option: use EarlyStopping only. Optuna will prune based on the
        # returned metric value between trials (MedianPruner / HyperbandPruner).
        # If you want step-level pruning, use a custom callback (see note below).
        early_stopping = EarlyStopping(
            monitor="valid_sindyzdot_loss",
            min_delta=0.0002,
            patience=5,
            mode="min",
            check_on_train_epoch_end=False,
        )

        trainer = L.Trainer(
            max_epochs=30,                  # shorter budget per trial
            log_every_n_steps=1,
            accelerator="gpu",
            devices=[gpu_id],               # pin trial to one GPU
            default_root_dir=f"{LOG_ROOT}/trial_{trial.number}",
            callbacks=[early_stopping],
            enable_progress_bar=False,      # suppress per-trial bars
            logger=True,
        )

        trainer.fit(sindy_sz, train_loader, valid_loader)

        # Return the monitored metric; Lightning stores it in callback_metrics.
        val_loss = trainer.callback_metrics.get("valid_sindyzdot_loss")
        if val_loss is None:
            # Training may have been stopped before any validation step logged
            # this key — return a large sentinel so Optuna deprioritises the trial.
            return float("inf")
        return float(val_loss)

    finally:
        gpu_queue.put(gpu_id)           # always release GPU
        del sindy_sz, trainer
        torch.cuda.empty_cache()


def main():
    # Use a multiprocessing-safe queue to hand out GPU slots.
    # Manager().Queue() is picklable and safe across forked processes.
    manager = mp.Manager()
    gpu_queue = manager.Queue()
    for i in range(N_GPUS):
        gpu_queue.put(i)

    study = optuna.create_study(
        storage=DB_PATH,
        study_name="sindy_sz_search",
        load_if_exists=True,
        direction="minimize",
        # HyperbandPruner operates at the trial level between epochs —
        # works fine even without step-level pruning callbacks.
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=3,
            max_resource=30,
            reduction_factor=3,
        ),
    )

    # n_jobs == N_GPUS: one worker process per GPU.
    # Each worker blocks on gpu_queue.get() until a slot is free.
    study.optimize(
        lambda trial: objective(trial, gpu_queue),
        n_trials=100,
        n_jobs=N_GPUS,
        gc_after_trial=True,        # free memory between trials
    )

    print("Best trial:")
    print(f"  value: {study.best_trial.value}")
    print(f"  params: {study.best_trial.params}")


if __name__ == "__main__":
    main()
