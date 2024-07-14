
import os
import os.path as osp
import glob
import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from dataset import Cityscapes
from cldm.logger import ImageLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from cldm.model import create_model, load_state_dict
from share import *


# Configs
resume_path = 'models/control_seg.ckpt'
model_conf_path = 'models/cldm_v15_small.yaml'
batch_size = 8
logger_freq = 200
learning_rate = 1e-5
sd_locked = True
only_mid_control = False

# Symlink Cityscapes dataset
if not osp.exists('training/target'):
    os.symlink(osp.realpath('../datasets/cityscapes/leftImg8bit'), 'training/target')

if not osp.exists('training/seg'):
    os.symlink(osp.realpath('../datasets/cityscapes/gtFine'), 'training/seg')

# Handling several GPUs
no_devices = torch.cuda.device_count()
gpus = list(range(no_devices))

# Logs dir
if not osp.exists('logs'):
    os.mkdir('logs')
num_dir = max([int(el) for el in os.listdir('logs/')], default=-1)
logs_dir = f'logs/{num_dir+1}'

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model(model_conf_path).cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# Misc
dataset = Cityscapes()
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
latest_checkpoint = ModelCheckpoint(filename='latest-{epoch}-{step}', every_n_epochs=10, save_top_k=-1)
trainer = pl.Trainer(gpus=gpus,
                     precision=32,
                     accelerator='gpu',
                     strategy='ddp',
                     callbacks=[logger, latest_checkpoint],
                     accumulate_grad_batches=16,
                     default_root_dir=logs_dir,
                     max_epochs=150)


# Train!
trainer.fit(model, dataloader)
