import config
import dataset_utils
from classification import LitConstrainedClassifier

import time
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

torch.manual_seed(config.random_seed)

# Training
train_data = dataset_utils.ImbalancedMNIST(
    props=config.props, 
    digits=config.digits, 
    transform=config.transform, 
    train=True, 
    download=True, 
    root='./data'
)
train_loader = DataLoader(
    dataset=train_data,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=config.num_workers,
    pin_memory=torch.cuda.is_available()
)

# Validation.
d = config.digits
val_props = [1/len(d) for i in range(len(d))] # perfectly balanced

val_data = dataset_utils.ImbalancedMNIST(
    props=val_props,
    digits=config.digits, 
    transform=config.transform, 
    train=False, 
    download=True, 
    root='./data'
)
val_loader = DataLoader(
    dataset=val_data,
    batch_size=config.val_batch_size,
    shuffle=False,
    num_workers=config.num_workers,
    pin_memory=torch.cuda.is_available()
)

# Initialize Model
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
    conv_kwargs=config.conv_kwargs,
)

# Training
early_stop = EarlyStopping(
    monitor='aug_lag',
    min_delta=config.stop_delta,
    patience=config.stop_patience,
    verbose=False,
    mode='min'
)
trainer = pl.Trainer(
    max_epochs=config.max_epochs,
    auto_scale_batch_size=config.auto_scale_batch_size,
    min_epochs=config.min_epochs,
    callbacks=[config.early_stop],
    checkpoint_callback=False,
    log_every_n_steps=config.log_every_n_steps, 
    flush_logs_every_n_steps=len(train_loader),
    log_gpu_memory=config.log_gpu_memory,
    gpus=config.gpus, # set to 0 if cpu use is prefered,
    auto_select_gpus=True,
    )

t = time.time() # Record time
trainer.fit(
    model, 
    train_dataloader=train_loader, 
    val_dataloaders=val_loader
    )
print("Training time:", time.time() - t)


