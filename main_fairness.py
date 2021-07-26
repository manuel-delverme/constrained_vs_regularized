import configs.config_fairness as config
from src.classification import LitConstrainedClassifier
import dataset_utils

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

seed_everything(config.random_seed)

# Training
train_data = dataset_utils.ImbalancedDataset(
    dataset_class=config.dataset_class,
    classes=config.classes,
    transform=config.transform,
    props=config.props,
    size=config.size,
    train=True,
    download=True,
    root='./data',
    )
train_loader = DataLoader(
    dataset=train_data,
    #batch_size=config.batch_size,
    batch_size=len(train_data),
    num_workers=config.num_workers,
    pin_memory=False, #torch.cuda.is_available()
)

"""
# Validation.
val_data = dataset_utils.ImbalancedDataset(
    dataset_class=config.dataset_class,
    classes=config.classes,
    transform=config.transform,
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
"""
# Initialize Model
constrained_kwargs=dict(
    optimizer_class=config.optimizer_class,
    le_levels=config.le_levels,
    eq_levels=config.eq_levels,
    le_names=config.le_names,
    eq_names=config.eq_names,
    model_lr=config.model_lr,
    dual_lr=config.dual_lr,
    log_constraints=config.log_constraints,
)

model = LitConstrainedClassifier(
    im_dim=config.im_dim,
    im_channels=config.im_channels,
    classes=config.classes,
    hid_dims=config.hid_dims,
    act_fun=config.act_fun,
    act_args=config.act_args,
    balanced_ERM=config.balanced_ERM,
    const_classes=config.const_classes,
    fairness=config.fairness,
    conv_channels=config.conv_channels,
    kernel_size=config.kernel_size,
    stride=config.stride,
    pool_k_size=config.pool_k_size,
    constrained_kwargs=constrained_kwargs,
)

# Training
early_stop = EarlyStopping(
    monitor='Lagrangian',
    min_delta=config.stop_delta,
    patience=config.stop_patience,
    verbose=False,
    mode='min',
    strict=True,
)

trainer = pl.Trainer(
    #logger=config.tensorboard,
    logger=WandbLogger(),
    max_epochs=config.max_epochs,
    auto_scale_batch_size=config.auto_scale_batch_size,
    min_epochs=config.min_epochs,
    callbacks=[early_stop],
    checkpoint_callback=False,
    log_every_n_steps=config.log_every_n_steps,
    flush_logs_every_n_steps=len(train_loader),
    log_gpu_memory=config.log_gpu_memory,
    gpus=config.gpus, # set to 0 if cpu use is prefered,
    auto_select_gpus=True,
    )

trainer.fit(
    model,
    train_dataloader=train_loader,
    #val_dataloaders=val_loader # for early stopping
    )