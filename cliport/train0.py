"""Main training script."""

import os
from pathlib import Path
import pdb

import torch
from cliport import agents
from cliport.dataset import RavensDataset, RavensMultiTaskDataset

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import WandbLogger

from hydra.core.global_hydra import GlobalHydra
from torch.utils.data import Subset
GlobalHydra.instance().clear()


# class SaveCallback(Callback):
#     def on_batch_end(self, trainer, pl_module):
#         train_loss = self.trainer.callback_metric['tr/loss']
#         steps = f'{self.trainer.global_step: 05d}'
#         filename = = f"steps={steps}-train_loss={val_loss:0.8f}.ckpt"


@hydra.main(version_base=None, config_path="./cfg", config_name='train_hydra')
def main(cfg: DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))
    # Logger
    wandb_logger = WandbLogger(name=cfg['tag']) if cfg['train']['log'] else None
    # Checkpoint saver
    # pdb.set_trace()
    hydra_dir = Path(os.getcwd())
    checkpoint_path = os.path.join(cfg['train']['train_dir'], 'checkpoints')
    last_checkpoint_path = os.path.join(checkpoint_path, 'last.ckpt')
    last_checkpoint = last_checkpoint_path if os.path.exists(last_checkpoint_path) and cfg['train']['load_from_last_ckpt'] else None
    # only use callback to save the best model, when monitor is the best?
    checkpoint_callback = ModelCheckpoint(
        monitor=cfg['wandb']['saver']['monitor'],
        mode='min',
        dirpath=checkpoint_path,
        filename='best',
        save_top_k=1,
        save_last=True
    )

    # Trainer
    max_epochs = cfg['train']['n_steps'] // cfg['train']['n_demos']  # 200
    trainer = Trainer(
        devices=cfg['train']['gpu'],
        accelerator='gpu',
        fast_dev_run=cfg['debug'],
        logger=wandb_logger,
        callbacks=checkpoint_callback,
        max_epochs=max_epochs,
        check_val_every_n_epoch=4,   # 200 // 50,
        log_every_n_steps=50, # how often to add logging rows(does not write to disk), default: 50
        default_root_dir=checkpoint_path
    )
    print('The max epochs: {}, check_val_every_n_epoch: {}'.format(max_epochs, 200 // 50))  # 201, 4

    # Resume epoch and global_steps
    if last_checkpoint:
        print(f"Resuming: {last_checkpoint}")

        # last_ckpt = torch.load(last_checkpoint)
        # trainer.current_epoch = last_ckpt['epoch']
        # trainer.global_step = last_ckpt['global_step']
        # del last_ckpt

    # Config
    data_dir = cfg['train']['data_dir']
    task = cfg['train']['task']
    agent_type = cfg['train']['agent']
    n_demos = cfg['train']['n_demos']
    n_val = cfg['train']['n_val']
    name = '{}-{}-{}'.format(task, agent_type, n_demos)

    # Datasets
    dataset_type = cfg['dataset']['type']
    if 'multi' in dataset_type:
        train_ds = RavensMultiTaskDataset(data_dir, cfg, group=task, mode='train', n_demos=n_demos, augment=True) # 不要数据增强，True改成False
        val_ds = RavensMultiTaskDataset(data_dir, cfg, group=task, mode='val', n_demos=n_val, augment=False)
    else:
        train_ds = RavensDataset(os.path.join(data_dir, '{}-train'.format(task)), cfg, n_demos=n_demos, augment=True) # 需要数据增强
        val_ds = RavensDataset(os.path.join(data_dir, '{}-val'.format(task)), cfg, n_demos=n_val, augment=False)

    # Initialize agent
    val_ds = Subset(val_ds,range(5))
    agent = agents.names[agent_type](name, cfg, train_ds, val_ds)

    # Main training loop
    trainer.fit(agent, ckpt_path='last')

if __name__ == '__main__':
    main()
