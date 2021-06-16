import argparse
import math
import os

import experiment_buddy

batch_size = 64
test_batch_size = 1000
epochs = 1
lr = 1.0
gamma = 0.7
no_cuda = False
dry_run = False
seed = 1
log_interval = 10
save_model = False

experiment_buddy.register(locals())

################################################################
# Derivative parameters
################################################################
tensorboard = experiment_buddy.deploy(host='', sweep_yaml="")
# tensorboard = experiment_buddy.deploy(host='mila', sweep_yaml="")
