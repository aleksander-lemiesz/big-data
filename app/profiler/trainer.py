import torch
import os
import pytorch_lightning as pl

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.loggers import TensorBoardLogger

from dataloaders.SRADataloader import SRADataloader
from app.profiler.classifiers.SexClassifier import SexClassifier
from app.profiler.classifiers.AgeClassifier import AgeClassifier
from app.profiler.classifiers.RaceClassifier import RaceClassifier

# a01 val
# epoch, step = 6, 32436
# r02 val
# epoch, step = 13, 70278
# s02 val
# epoch, step = 9, 48654

if __name__ == '__main__':
    experiment: str = 'a01'  # [sra][01-99]
    mode: str = 'test'  # 'train' or 'test'
    load_set: str = 'global'  # 'global' or 'val'
    epoch, step = 6, 32436  # choose correct if loading from checkpoint

    # optimisation
    device = 'cpu'  # 'gpu' or 'cpu'
    no_workers = min(4, os.cpu_count()//2)
    # using only 50% of capabilities
    if device == 'gpu':
        torch.cuda.set_per_process_memory_fraction(0.5, 0)

    data_filename = 'data_full.csv'
    load = os.path.join('app/profiler/checkpoints', f'SRA_{experiment}',
                        load_set, f'epoch={epoch}-step={step}.ckpt')

    if experiment.startswith('s'):
        dl = SRADataloader('s', filename=data_filename, no_workers=no_workers)
        model = SexClassifier(learning_rate=1e-4, weights=dl.get_weights())
    elif experiment.startswith('r'):
        dl = SRADataloader('r', filename=data_filename, no_workers=no_workers)
        model = RaceClassifier(learning_rate=1e-4, weights=dl.get_weights())
    elif experiment.startswith('a'):
        dl = SRADataloader('a', filename=data_filename, no_workers=no_workers)
        model = AgeClassifier(learning_rate=1e-4, weights=dl.get_weights())
    else:
        raise Exception(f'Invalid experiment: {experiment}')

    checkpoint_val_cb = ModelCheckpoint(
        dirpath=f'app/profiler/checkpoints/SRA_{experiment}/val',
        save_top_k=3,
        monitor='val_acc',
        mode='max'
    )
    checkpoint_cb = ModelCheckpoint(
        dirpath=f'app/profiler/checkpoints/SRA_{experiment}/global',
        save_top_k=3,
        monitor='epoch',
        mode='max'
    )
    early_stop_cb = EarlyStopping(
        monitor='val_acc',
        min_delta=0.0001,
        patience=5,
        verbose=False,
        mode='max'
    )

    progress_bar = RichProgressBar(
        leave=True,
        theme=RichProgressBarTheme(
            description="green_yellow",
            progress_bar="green1",
            progress_bar_finished="green1",
            progress_bar_pulse="#6206E0",
            batch_progress="green_yellow",
            time="grey82",
            processing_speed="grey82",
            metrics="grey82",
        )
    )

    logger = TensorBoardLogger(save_dir='app/profiler/logs', name=f'SRA_{experiment}')
    trainer = pl.Trainer(
        max_epochs=1000,
        accelerator=device,
        devices=1,
        logger=logger,
        # auto_scale_batch_size='binsearch',
        callbacks=[
            progress_bar,
            # early_stop_cb,
            checkpoint_cb,
            checkpoint_val_cb
        ]
    )

    trainer.tune(model)

    if mode == 'train':
        try:
            trainer.fit(model, dl, ckpt_path=load)
        except (PermissionError, FileNotFoundError):
            trainer.fit(model, dl)
    elif mode == 'test':
        trainer.test(model, dl, ckpt_path=load)
