import configs.config_fairness as config
from src.classification import LitConstrainedClassifier
import dataset_utils

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

seed_everything(config.random_seed)

# Training
over_prop = (1. - config.under_prop * len(config.under_classes)) / (len(config.classes) - len(config.under_classes))

props = [config.under_prop if c in config.under_classes else over_prop for c in config.classes]
print("Proportions:", props)

train_data = dataset_utils.ImbalancedDataset(
    dataset_class=config.dataset_class,
    classes=config.classes,
    transform=config.transform,
    device=config.device,
    props=props,
    size=config.size,
    train=True,
    download=True,
    root='./data',
    )
print(train_data.imbal_cl_counts)

b_size = config.batch_size if config.batch_size is not None else len(train_data)
train_loader = DataLoader(
    dataset=train_data,
    batch_size=b_size,
    shuffle=False,
    num_workers=config.num_workers,
    pin_memory=False#torch.cuda.is_available() # introduces weird warnings
)

# Validation.
val_data = dataset_utils.ImbalancedDataset(
    dataset_class=config.dataset_class,
    classes=config.classes,
    transform=config.transform,
    device=config.device,
    props=None,
    train=False,
    download=True,
    root='./data',
)
val_loader = DataLoader(
    dataset=val_data,
    batch_size=len(val_data),
    shuffle=False,
    num_workers=config.num_workers,
    pin_memory=False#torch.cuda.is_available()
)

# Re seed to make model initialization equivalent
seed_everything(config.random_seed)

# Initialize Model
constrained_kwargs=dict(
    optimizer_class=config.optimizer_class,
    is_constrained=config.is_constrained,
    le_levels=config.le_levels,
    eq_levels=config.eq_levels,
    le_names=config.le_names,
    eq_names=config.eq_names,
    model_lr=config.model_lr,
    dual_lr=config.dual_lr,
    augmented_lagrangian_coefficient = config.augmented_lagrangian_coefficient,
    log_constraints=config.log_constraints,
    metrics=config.metrics,
)

model = LitConstrainedClassifier(
    im_dim=config.im_dim,
    im_channels=config.im_channels,
    classes=config.classes,
    hid_dims=config.hid_dims,
    act_fun=config.act_fun,
    act_args=config.act_args,
    balanced_ERM=config.balanced_ERM,
    fairness=config.fairness,
    conv_channels=config.conv_channels,
    kernel_size=config.kernel_size,
    stride=config.stride,
    pool_k_size=config.pool_k_size,
    constrained_kwargs=constrained_kwargs,
)

print(model)

# Training
# JC ToDo: clean the early stop
is_constrained = config.le_levels is not None or config.eq_levels is not None
to_monitor = 'val/lagrangian' if is_constrained else "val/ERM"
early_stop = EarlyStopping(
    monitor=to_monitor,
    min_delta=config.stop_delta,
    patience=config.stop_patience,
    verbose=False,
    mode='min',
    strict=True,
)

gpus = -1 if config.device == "cuda" else 0
trainer = pl.Trainer(
    #logger=config.tensorboard,
    logger=WandbLogger(),
    max_epochs=config.max_epochs,
    auto_scale_batch_size=config.auto_scale_batch_size,
    min_epochs=config.min_epochs,
    #callbacks=[early_stop],
    checkpoint_callback=False,
    log_every_n_steps=config.log_every_n_steps,
    flush_logs_every_n_steps=len(train_loader),
    log_gpu_memory=config.log_gpu_memory,
    gpus=gpus,
    auto_select_gpus=gpus!=0,
    distributed_backend='ddp',
    #profiler="simple",
    )

trainer.fit(
    model,
    train_dataloader=train_loader,
    #val_dataloaders=val_loader # for early stopping,
    )