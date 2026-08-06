# main.py
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import torch
import torch.utils.data as data
import lightning as L
from lightning.pytorch.callbacks import Callback, EarlyStopping
from torch.utils.data import DataLoader
import multiprocessing as mp
import logging
import os

from datasets import RawBicoherenceSequenceDataset
from model import SINDySz, ConvSINDyEncoder, ConvSINDyDecoder
from fullres_autoencoder import FullResAutoencoder


class OptunaProgressCallback(Callback):
    """Update Optuna trial user attributes with live epoch and batch progress.

    Writes two attributes after every training batch so the Optuna dashboard
    can show where a running trial is within its training budget:
      - ``current_epoch``:    integer epoch index (0-based, matching Lightning).
      - ``epoch_pct``:        float in [0, 100] — percentage of the current
                              epoch's batches that have been processed.

    Attributes are also written at epoch start (``epoch_pct = 0.0``) so the
    dashboard shows an update immediately when a new epoch begins, even if
    batches are slow.
    """

    def __init__(self, trial: optuna.Trial, max_epochs: int):
        super().__init__()
        self.trial = trial
        self.max_epochs = int(max_epochs)
        # Cached total number of training batches per epoch; resolved once from
        # the first on_train_epoch_start call so we don't rely on the dataloader
        # being finite-length at construction time.
        self._batches_per_epoch: int | None = None

    def _resolve_batches_per_epoch(self, trainer: L.Trainer) -> int:
        """Return total training batches per epoch, or a fallback of 1."""
        try:
            n = trainer.num_training_batches
            if n is not None and n > 0:
                return int(n)
        except Exception:
            pass
        return 1

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._batches_per_epoch = self._resolve_batches_per_epoch(trainer)
        epoch = int(trainer.current_epoch)
        self.trial.set_user_attr("current_epoch", epoch)
        self.trial.set_user_attr("max_epochs", self.max_epochs)
        self.trial.set_user_attr("epoch_pct", 0.0)

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        total = self._batches_per_epoch or self._resolve_batches_per_epoch(trainer)
        if total and total > 0:
            pct = round(100.0 * (batch_idx + 1) / total, 1)
        else:
            pct = 0.0
        epoch = int(trainer.current_epoch)
        self.trial.set_user_attr("current_epoch", epoch)
        self.trial.set_user_attr("epoch_pct", pct)


class OptunaPruningCallback(Callback):
    """Report validation loss to Optuna after each epoch and prune if needed.

    Calls ``trial.report(value, step)`` so that the HyperbandPruner can act
    on per-epoch intermediate results rather than only between completed trials.
    Also records secondary metrics as trial user attributes for dashboard
    visibility (they do not affect the pruning objective).
    """

    def __init__(self, trial: optuna.Trial, monitor: str = "valid_sindyzdot_loss"):
        super().__init__()
        self.trial = trial
        self.monitor = monitor

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        metrics = trainer.callback_metrics

        # Primary metric: drives pruning decision.
        val = metrics.get(self.monitor)
        if val is None:
            return
        self.trial.report(float(val), step=trainer.current_epoch)

        # Secondary metrics: stored as user attrs for richer dashboard display.
        for attr_key, metric_key in (
            ("valid_R2_recon_last",    "valid_R2_recon"),
            ("valid_R2_xdot_last",     "valid_R2_xdot"),
            ("valid_R2_zdot_last",     "valid_R2_zdot"),
            ("valid_recon_loss_last",  "valid_recon_loss"),
            ("valid_sindy_loss_last",  "valid_sindy_loss"),
            ("valid_decoder_loss_last","valid_decoder_loss"),
        ):
            v = metrics.get(metric_key)
            if v is not None:
                self.trial.set_user_attr(attr_key, float(v))

        if self.trial.should_prune():
            raise optuna.TrialPruned(
                f"Trial {self.trial.number} pruned at epoch {trainer.current_epoch} "
                f"({self.monitor}={float(val):.6g})"
            )


N_GPUS = int(os.getenv("N_GPUS", "1"))
DATA_FILE = "/app/Data/WR/WR5_Run4.hdf5"
SAMPLE_RATE = 5000
LOG_ROOT = "/app/Repos/pytorchSINDySz/lightning_logs/optuna"
DB_PATH = "sqlite:////app/Data/WR/optuna_sindy.db"


def build_conv_masked_ae(height: int, width: int, latent_dim: int):
    shared_ae = FullResAutoencoder(height=height, width=width, latent_dim=latent_dim)
    encoder = ConvSINDyEncoder(height=height, width=width, latent_dim=latent_dim, ae=shared_ae)
    decoder = ConvSINDyDecoder(height=height, width=width, latent_dim=latent_dim, ae=shared_ae)
    return encoder, decoder


def make_dataloaders(time_dim, map_time_step, batch_size):
    dataset = RawBicoherenceSequenceDataset(
        data_file=DATA_FILE,
        seq_len=time_dim,
        epoch_size=map_time_step,
        segment_seconds=0.75,
        segment_overlap=0.5,
        f_max=25.0,
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

    # Per-trial file handler for gradient debug output from model.py.
    # Writing to a file (rather than stdout) is safe in multiprocessing worker
    # subprocesses where stdout can be closed during trial teardown.
    trial_log_dir = f"{LOG_ROOT}/trial_{trial.number}"
    os.makedirs(trial_log_dir, exist_ok=True)
    _trial_fh = logging.FileHandler(
        f"{trial_log_dir}/grad_debug.log", mode="a", delay=False
    )
    _trial_fh.setLevel(logging.DEBUG)
    _trial_fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    _model_logger = logging.getLogger("model")
    _model_logger.setLevel(logging.DEBUG)
    _model_logger.addHandler(_trial_fh)

    try:
        # ── Hyperparameters to search ─────────────────────────────────────────
        lr            = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        sindy_lr      = trial.suggest_float("sindy_lr", 1e-5, 1e-2, log=True)
        decoder_lr    = trial.suggest_float("decoder_lr", 1e-5, 1e-2, log=True)
        latent_features = trial.suggest_categorical("latent_features", [6, 9, 12, 16])
        poly_order    = trial.suggest_categorical("poly_order", [1, 2, 3])
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
        # OptunaPruningCallback calls trial.report() after every validation epoch
        # so the HyperbandPruner can act on per-epoch intermediate results.
        # (The standard PyTorchLightningPruningCallback only works with automatic
        # optimization; our manual dual-optimizer loop requires this custom one.)
        pruning_cb = OptunaPruningCallback(trial, monitor="valid_sindyzdot_loss")
        progress_cb = OptunaProgressCallback(trial, max_epochs=30)
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
            callbacks=[early_stopping, pruning_cb, progress_cb],
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

        # ── Post-fit user attributes ───────────────────────────────────────────
        # Store interpretable attributes that survive in the Optuna DB and are
        # visible in the dashboard, without affecting the pruning objective.
        metrics = trainer.callback_metrics

        # Decomposed final validation losses.
        for attr_key, metric_key in (
            ("final_valid_recon_loss",   "valid_recon_loss"),
            ("final_valid_sindy_loss",   "valid_sindy_loss"),
            ("final_valid_decoder_loss", "valid_decoder_loss"),
            ("final_valid_R2_recon",     "valid_R2_recon"),
            ("final_valid_R2_xdot",      "valid_R2_xdot"),
            ("final_valid_R2_zdot",      "valid_R2_zdot"),
        ):
            v = metrics.get(metric_key)
            if v is not None:
                trial.set_user_attr(attr_key, float(v))

        # SINDy weight sparsity: fraction of coefficients thresholded to zero.
        with torch.no_grad():
            w = sindy_sz.sindy_model.SINDy_predict.weight
            sparsity = float((w.abs() < 1e-8).float().mean().item())
        trial.set_user_attr("sindy_sparsity_final", sparsity)

        # Convenience copies of key hyperparams as user attrs so they appear
        # alongside the metrics in the dashboard without needing to cross-reference
        # the params table.
        trial.set_user_attr("latent_features", latent_features)
        trial.set_user_attr("poly_order", poly_order)
        trial.set_user_attr("time_dim", time_dim)

        return float(val_loss)

    finally:
        gpu_queue.put(gpu_id)           # always release GPU
        del sindy_sz, trainer
        torch.cuda.empty_cache()
        # Remove and close the per-trial file handler so it doesn't accumulate
        # across repeated calls in the same worker process.
        _model_logger.removeHandler(_trial_fh)
        _trial_fh.close()


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
