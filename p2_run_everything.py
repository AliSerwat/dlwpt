# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + language="bash"
# #!/bin/bash
#
# # Clear pip cache
# pip cache purge
#
# # Clear conda cache
# conda clean --all -y
#
# # Clear apt and apt-get cache
# sudo apt-get clean
# sudo apt-get autoclean
# sudo apt-get autoremove -y
#
# # Clear snapd cache
# sudo rm -rf /var/cache/snapd/
#
# # Clear git untracked files
# # Note: Be cautious with this command, as it will remove untracked files
# git clean -fdx
#
# # Clear git credentials cache
# git credential-cache exit
#
# # Clear npm cache
# npm cache clean --force
#
# # Clear yarn cache
# yarn cache clean
#
# # Clear Docker cache
# docker system prune -a --volumes -f
#
# # Clear systemd journal logs
# sudo journalctl --vacuum-time=1s
#
# # Clear temporary files
# sudo rm -rf /tmp/*
# sudo rm -rf /var/tmp/*
#
# # Clear user cache files
# rm -rf ~/.cache/*
#
# # Clear trash files
# rm -rf ~/.local/share/Trash/*
#
# # Clear thumbnail cache
# rm -rf ~/.cache/thumbnails/*
#
# # Clean system logs
# sudo find /var/log -type f -delete
#
# # Clean apt archive cache
# sudo rm -rf /var/cache/apt/archives/*
#
# -

import os
os.chdir('/teamspace/studios/this_studio/dlwpt-code')


# +
import datetime

from util.util import importstr
from util.logconf import logging
log = logging.getLogger('nb')

# -

def run(app, *argv):
    argv = list(argv)
    argv.insert(0, '--num-workers=4')  # <1>
    log.info("Running: {}({!r}).main()".format(app, argv))
    
    app_cls = importstr(*app.rsplit('.', 1))  # <2>
    app_cls(argv).main()
    
    log.info("Finished: {}.{!r}).main()".format(app, argv))



# +
import os
import shutil

# clean up any old data that might be around.
# We don't call this by default because it's destructive, 
# and would waste a lot of time if it ran when nothing 
# on the application side had changed.
def cleanCache():
    shutil.rmtree('data-unversioned/cache')
    os.mkdir('data-unversioned/cache')

# cleanCache()



# -

training_epochs = 20
experiment_epochs = 10
final_epochs = 50


# +

training_epochs = 2
experiment_epochs = 2
final_epochs = 5
seg_epochs = 10

# -

# ## Chapter 11

# run('p2ch11.prepcache.LunaPrepCacheApp')


# run('p2ch11.training.LunaTrainingApp', '--epochs=1')


run('p2ch11.training.LunaTrainingApp', f'--epochs={experiment_epochs}')


# ## Chapter 12

run('p2ch12.prepcache.LunaPrepCacheApp')


run('p2ch12.training.LunaTrainingApp', '--epochs=1', 'unbalanced')


run('p2ch12.training.LunaTrainingApp', f'--epochs={training_epochs}', '--balanced', 'balanced')


run('p2ch12.training.LunaTrainingApp', f'--epochs={experiment_epochs}', '--balanced', '--augment-flip', 'flip')


run('p2ch12.training.LunaTrainingApp', f'--epochs={experiment_epochs}', '--balanced', '--augment-offset', 'offset')


run('p2ch12.training.LunaTrainingApp', f'--epochs={experiment_epochs}', '--balanced', '--augment-scale', 'scale')


run('p2ch12.training.LunaTrainingApp', f'--epochs={experiment_epochs}', '--balanced', '--augment-rotate', 'rotate')


run('p2ch12.training.LunaTrainingApp', f'--epochs={experiment_epochs}', '--balanced', '--augment-noise', 'noise')


run('p2ch12.training.LunaTrainingApp', f'--epochs={training_epochs}', '--balanced', '--augmented', 'fully-augmented')


# ## Chapter 13

run('p2ch13.prepcache.LunaPrepCacheApp')


run('p2ch13.training.LunaTrainingApp', f'--epochs={final_epochs}', '--balanced', '--augmented', 'final-cls')


run('p2ch13.train_seg.LunaTrainingApp', f'--epochs={seg_epochs}', '--augmented', 'final-seg')


# ## Chapter 14

run('p2ch14.prepcache.LunaPrepCacheApp')


run('p2ch14.training.ClassificationTrainingApp', f'--epochs=100', 'nodule-nonnodule')


run('p2ch14.training.ClassificationTrainingApp', f'--epochs=40', '--malignant', '--dataset=MalignantLunaDataset',
    '--finetune=''data/part2/models/cls_2020-02-06_14.16.55_final-nodule-nonnodule.best.state',
    'finetune-head')


run('p2ch14.training.ClassificationTrainingApp', f'--epochs=40', '--malignant', '--dataset=MalignantLunaDataset',
    '--finetune=''data/part2/models/cls_2020-02-06_14.16.55_final-nodule-nonnodule.best.state',
    '--finetune-depth=2',
    'finetune-depth2')


run('p2ch14.nodule_analysis.NoduleAnalysisApp', '--run-validation')


run('p2ch14.nodule_analysis.NoduleAnalysisApp', '--run-validation', '--malignancy-path')





