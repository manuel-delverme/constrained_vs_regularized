import experiment_buddy
import torch_constrained

import torch
from torchvision import transforms, datasets
import torchmetrics as tm
from torchmetrics import MetricCollection

random_seed = 5
torch.cuda.is_available()

# -------------------------------------------------------- Data
# Dataset
dataset_class = datasets.MNIST

classes = [1,4,7,9]
under_prop = 0.7
under_classes = [1]
size = 5000

# Image details and DataLoader
im_dim = 28
im_channels = 1
transform=transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)), # CIFAR10
        transforms.Normalize((0.1307,), (0.3081,)), # MNIST
        ])
batch_size = None
num_workers = 0

# ------------------------------------------------------ Application

# MLP layers
hid_dims = None
act_fun = None
act_args = []

# Conv layers
conv_channels = None
kernel_size = None
stride = None
pool_k_size = None

# Objective function and constraints
balanced_ERM = False
fairness = False

# --------------------------------------------------- torch constrained
# Solve via ERM or constrained formulation?
is_constrained = True

# (Possibly) constrained Objectives
eq_levels = None
eq_names = None
le_levels = [0.01, None, None, None]
le_names = ["CE_class_1","CE_class_4","CE_class_7","CE_class_9"]
model_lr = 0.005
dual_lr = 1e-2
log_constraints = True
augmented_lagrangian_coefficient = False # false if not used
optimizer_class = torch_constrained.ExtraAdam

#----------------------------------------------------- training
stop_delta = 1e-6
stop_patience = 15

max_epochs = 500
auto_scale_batch_size = False
min_epochs = 10
log_every_n_steps = 1
log_gpu_memory = False
device = "cuda"

# -------------------------------------------------- Metrics
metrics = MetricCollection({
        "metrics/accuracy": tm.Accuracy(num_classes=len(classes), average="micro"), # not balanced
        "metrics/balanced_accuracy": tm.Accuracy(num_classes=len(classes), average="macro"), # balanced
        "metrics/accuracy_per_class": tm.Accuracy(num_classes=len(classes), average="none"), # per class
})
# Imbalanced metric + None

# Experiment Buddy
experiment_buddy.register(locals())
tensorboard = experiment_buddy.deploy(wandb_kwargs={"tags": ["fix_data_size", "imbalance", "XG"]})
# sweep_yaml = "" # for cluster sweeps
# wandb_kwargs={"group": "GPU usage"} # for group runs
# wandb_kwargs={"tags": ["GPU usage"]} # for tags