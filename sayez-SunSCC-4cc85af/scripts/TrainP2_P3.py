from bioblue.transforms import *
from pathlib import Path

import bioblue
import pytorch_lightning as pl
import torch
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
# %matplotlib ipympl
import ipywidgets as widget
import os
from omegaconf import DictConfig, OmegaConf, open_dict
from hydra.utils import instantiate

import shutil
from datetime import datetime

from typing import Mapping
from tqdm.notebook import tqdm
import wandb


from bioblue.nb.load import load_from_dir, load_from_dir2, load_from_cfg

import argparse


def main(args):

    run_dir = Path(args.run_dir)

    config, model, dm, trainer = load_from_dir2(
                                    run_path= run_dir,
                                    load_trainer=False,
                                    override = [
                                    ]
                                )
    
    trained_parts = config.model.parts_to_train
    trained_parts_str = str(config.model.parts_to_train).upper().replace("[","").replace("]","").replace("'","").replace(", ", '_')
    print(trained_parts_str)

    all_overrides = {
                #   "scheduler":{"_target_": "torch.optim.lr_scheduler.MultiStepLR",
                #                "milestones": [2, 15 , 25],
                #                 "gamma": 0.1},
                #   "scheduler_interval": 'epoch',
                #   "wandb":{"project":"McIntosh_Fast_Cascade_DataAugFlipRotate_SplitTraining"}
                  "wandb":{"project":"McIntosh_Fast_NoCascade_DataAugFlipRotate_SplitTraining"}
    }

    shutil.copy(os.path.join(run_dir,f'models/last.ckpt'), os.path.join(run_dir,f'models/{trained_parts_str}.ckpt'))
    everything = torch.load(os.path.join(run_dir,f'models/{trained_parts_str}.ckpt'))
    model.load_state_dict(everything['state_dict'])
    torch.save(model.state_dict(), 
           os.path.join(run_dir,f'models/{trained_parts_str}.pth'))
    


    ####### TRAIN MLP2
    print('------------')
    print('Training MLP2')
    print('------------')
    print('Configuring MLP2')

    conf2 = deepcopy(config)

    conf2.model.parts_to_train = ['MLP2']

    conf2.trainer.max_epochs=100
    conf2.dataset.char_to_balance = 'class2'

    # conf2.module.lr=1e-8
    # conf2.module.scheduler=all_overrides["scheduler"]
    # conf2.module.scheduler_interval=all_overrides["scheduler_interval"]
    conf2.logger[0]['project']=all_overrides["wandb"]['project']

    module, datamodule = load_from_cfg(conf2, recursive=False)

    print('Configuring Trainer')
    callbacks = []
    if isinstance(conf2.callbacks, Mapping):
        conf2.callbacks = [cb for cb in conf2.callbacks.values()]
    for callback in conf2.callbacks:
        if callback['_target_'] == 'bioblue.callback.WandBCallback':
            print(type(callback))
    #         callback['cfg'] = conf2
            with open_dict(callback):
                callback.cfg = conf2
            continue
        callback = instantiate(callback, _recursive_=False)
        callback.conf = conf2  # FIXME : ugly hack
        callbacks.append(callback)

    trainer: pl.Trainer = instantiate(
        conf2.trainer,
        logger=conf2.logger,
        default_root_dir=".",
        callbacks=callbacks,
        _recursive_=True,
        _convert_="all",
    #     strategy="ddp",
        precision=16
    )
    trainer.tune(module, datamodule=datamodule)

    print('loading model')
    module.load_state_dict(torch.load(run_dir / f"models/{trained_parts_str}.pth"))
    
    print('Training MLP2')
    trainer.fit(model=module,datamodule=datamodule)

    print('Saving MLP2')
    trained_parts2 = conf2.model.parts_to_train
    print(trained_parts2)
    trained_parts_str2 = str(trained_parts2).upper().replace("[","").replace("]","").replace("'","").replace(", ", '_')
    trained_parts_str2 = trained_parts_str + '_' + trained_parts_str2
    print(trained_parts_str2)


    shutil.copy(os.path.join('.',f'models/last.ckpt'), os.path.join(run_dir,f'models/{trained_parts_str2}.ckpt'))
    everything = torch.load(os.path.join(run_dir,f'models/{trained_parts_str2}.ckpt'))
    model.load_state_dict(everything['state_dict'])
    torch.save(model.state_dict(), 
           os.path.join(run_dir,f'models/{trained_parts_str2}.pth'))
    


    ####### TRAIN MLP3
    print('------------')
    print('Training MLP2')
    print('------------')
    print('Configuring MLP2')


    conf3 = deepcopy(config)

    conf3.model.parts_to_train = ['MLP3']

    conf3.trainer.max_epochs=100
    conf3.dataset.char_to_balance = 'class3'

    # conf3.module.lr=1e-8
    # conf3.module.scheduler=all_overrides["scheduler"]
    # conf3.module.scheduler_interval=all_overrides["scheduler_interval"]
    conf3.logger[0]['project']=all_overrides["wandb"]['project']

    module, datamodule = load_from_cfg(conf3, recursive=False)

    print('Configuring Trainer')
    callbacks = []
    if isinstance(conf3.callbacks, Mapping):
        conf3.callbacks = [cb for cb in conf3.callbacks.values()]
    for callback in conf3.callbacks:
        if callback['_target_'] == 'bioblue.callback.WandBCallback':
            print(type(callback))
    #         callback['cfg'] = conf3
            with open_dict(callback):
                callback.cfg = conf3
            continue
        callback = instantiate(callback, _recursive_=False)
        callback.conf = conf3  # FIXME : ugly hack
        callbacks.append(callback)

    trainer: pl.Trainer = instantiate(
        conf3.trainer,
        logger=conf3.logger,
        default_root_dir=".",
        callbacks=callbacks,
        _recursive_=True,
        _convert_="all",
    #     strategy="ddp",
        precision=16
    )
    trainer.tune(module, datamodule=datamodule)

    print('loading model')
    module.load_state_dict(torch.load(run_dir / "models/ENCODER_MLP1_MLP2.pth"))
    
    print('Training MLP3')
    trainer.fit(model=module,datamodule=datamodule)

    print('Saving MLP3')
    trained_parts3 = conf3.model.parts_to_train
    print(trained_parts3)
    trained_parts_str3 = str(trained_parts3).upper().replace("[","").replace("]","").replace("'","").replace(", ", '_')
    trained_parts_str3 = trained_parts_str2 + '_' + trained_parts_str3
    print(trained_parts_str3)

    shutil.copy(os.path.join('.',f'models/last.ckpt'), os.path.join(run_dir,f'models/{trained_parts_str3}.ckpt'))
    everything = torch.load(os.path.join(run_dir,f'models/{trained_parts_str3}.ckpt'))
    model.load_state_dict(everything['state_dict'])
    torch.save(model.state_dict(), 
            os.path.join(run_dir,f'models/{trained_parts_str3}.pth'))


    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, default='.', help='run')
    args = parser.parse_args()

    cwd = os.getcwd()
    print(cwd)
    if not os.path.exists(args.run_dir+'/P2_P3'):
        os.makedirs(args.run_dir+'/P2_P3')
    os.chdir(args.run_dir+'/P2_P3')
    new_cwd = os.getcwd()
    print(new_cwd)

    main(args)
