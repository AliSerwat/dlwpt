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

# +
import torch

# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    # Define a tensor and move it to GPU
    device = torch.device("cuda")
    x = torch.rand(3, 3).to(device)
    print(x)

    # Run your deep learning model training or inference here
    # This will utilize the GPU automatically if properly configured
else:
    print("CUDA (GPU) is not available on this system.")

# -

# # for updating Conda libraries and system packages via notebook

# +
# # %%bash

# # Update package lists for Ubuntu
# # echo "Updating package lists..."
# sudo apt-get update

# # Perform a distribution upgrade
# # echo "Performing a distribution upgrade..."
# sudo apt-get dist-upgrade -y

# # Upgrade all installed packages to the newest version
# # echo "Upgrading all installed packages..."
# sudo apt-get upgrade -y

# # Remove unnecessary packages and clean up
# # echo "Cleaning up unnecessary packages..."
# sudo apt-get autoremove -y
# sudo apt-get clean

# # Update Conda packages
# if command -v conda &> /dev/null; then
#     echo "Updating Conda packages..."
#     conda update --all -y
# else
#     echo "Conda is not installed. Skipping Conda update."
# fi

# # Update pip packages
# if command -v pip &> /dev/null; then
#     echo "Updating pip packages..."
#     pip list --outdated --format=freeze | grep -v '^\-e' | cut -d = -f 1 | xargs -n1 sudo pip install -U
# else
#     echo "pip is not installed. Skipping pip update."
# fi

# # echo "System update complete!"



# +
# import os
# import shutil

# def remove_files_except_specified(base_path, allowed_extensions):
#     """
#     Remove all files except those with specified extensions in the given directory and its subdirectories.

#     :param base_path: Path to the directory from which files will be removed.
#     :param allowed_extensions: List of extensions to keep.
#     """
#     for root, dirs, files in os.walk(base_path):
#         for file in files:
#             file_path = os.path.join(root, file)
#             file_ext = os.path.splitext(file)[1].lower()
#             if file_ext not in allowed_extensions:
#                 print(f"Removing {file_path}")
#                 os.remove(file_path)

# # Define the directories to clean
# directories_to_clean = [
#     '/teamspace/studios/this_studio/dlwpt-code/data-unversioned',
#     '/teamspace/studios/this_studio/dlwpt-code/data'
# ]

# # List of allowed extensions
# allowed_extensions = ['.mhd', '.raw', '.zip', '.csv', '.sh']

# # Iterate over each directory and apply the file removal logic
# # for directory in directories_to_clean:
# #     remove_files_except_specified(directory, allowed_extensions)



# +
from diskcache import FanoutCache
import shutil
import os
from pathlib import Path

repo_path = Path('/teamspace/studios/this_studio/dlwpt-code')
os.chdir(repo_path)


def clear_fanout_cache(scope_str):
    """
    Clear the contents of a FanoutCache for a given scope.

    :param scope_str: Scope string for the FanoutCache.
    """
    cache_path = f'data-unversioned/cache/{scope_str}'
    cache = FanoutCache(cache_path)
    cache.clear()
    cache.close()
    print(f"Cache for '{scope_str}' has been cleared.")


# Clear FanoutCache contents for a specific scope
# clear_fanout_cache('part2ch10_raw')



# +
import os
ext_set = set()
for root, dirs, files in os.walk('/teamspace/studios/this_studio/dlwpt-code/data-unversioned'):
    for file in files:
        ext_set.add(os.path.splitext(file)[1])

ext_list = list(ext_set)

print(ext_list)



# +
import os
from pathlib import Path

repo_path = Path('/teamspace/studios/this_studio/dlwpt-code')
os.chdir(repo_path)

import logging
import multiprocessing
import functools
import csv
import copy
import math
import argparse
import datetime
import sys
import glob

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.cuda
import torch.distributed as dist
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
from torch.optim import SGD, Adam
from collections import namedtuple
from IPython.display import display, HTML

from util.logconf import logging
from util.util import XyzTuple, xyz2irc, enumerateWithEstimate
from util.disk import getCache
from p2ch11.dsets import LunaDataset
from p2ch11.model import LunaModel
from GPUtil import GPUtil
import psutil

# Set up paths and logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
# Data path
data_path = (repo_path/'data-unversioned/part2/luna/')
os.listdir(data_path)

# -

# !conda install conda-forge::gputil


# Used for computeBatchLoss and logMetrics to index into metrics_tensor/metrics_a
METRICS_LABEL_NDX = 0
METRICS_PRED_NDX = 1
METRICS_LOSS_NDX = 2
METRICS_SIZE = 3



# +
# Create a sample DataFrame
df = pd.DataFrame({
    'A': range(1, 11),
    'B': range(11, 21),
    'C': range(21, 31),
    'D': range(31, 41)
})

# Set Pandas display options
pd.set_option('display.max_rows', 5)  # Set maximum number of rows to display
# Set maximum number of columns to display
pd.set_option('display.max_columns', 10)

# Define custom CSS for larger font size


def display_large(df):
    display(HTML(f"""
    <style>
    .dataframe td, .dataframe th {{
        font-size: 15px;
    }}
    </style>
    {df.to_html()}
    """))


# Display the DataFrame with larger font size
display_large(df)



# +

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

raw_cache = getCache('part2ch10_raw')



# +
csv_files = glob.glob(str(data_path/"*.csv"))

for file in csv_files:
    fname, _ = os.path.splitext(os.path.split(file)[1])
    df_name = f"{fname}_df"
    globals()[df_name] = pd.read_csv(file)
    print(df_name)

# -

# annotations = total number of nodules among "CANDIDATES"
annotations_df.head(1)



CandidInfoTuple = namedtuple(
    "CandidInfoTuple", ['is_nodule', 'uid', 'xyz_center', 'diameter'])



# candidates = total number of lumps
candidates_df.head(1)


@functools.lru_cache(1)
def get_candid_info_list(data_directory, require_on_disk=True):
    mhd_list = glob.glob(os.path.join(data_directory, 'subset*', '*.mhd'))
    downloaded = {os.path.splitext(os.path.split(mhd)[1])[
        0] for mhd in mhd_list}

    ann_spatial_data = {}
    with open(os.path.join(data_directory, 'annotations.csv'), 'r') as f:
        for row in list(csv.reader(f))[1:]:
            uid = row[0]
            ann_xyz_center = tuple(float(x) for x in row[1:4])
            diameter = float(row[4])
        ann_spatial_data.setdefault(uid, []).append(
            (ann_xyz_center, diameter)
        )

    candid_info_list = []
    with open(os.path.join(data_directory, 'candidates.csv')) as f:
        for row in list(csv.reader(f))[1:]:
            uid = row[0]
            if uid not in downloaded and require_on_disk:
                continue
            is_nodule = int(row[4])
            candid_xyz_center = tuple(float(x) for x in row[1:4])
            candid_diameter = 0.0
            for ann_xyz_center, ann_diameter in ann_spatial_data.get(uid, []):
                for i in range(3):

                    intercenter_diameter = abs(
                        candid_xyz_center[i] - ann_xyz_center[i])
                    if intercenter_diameter > ann_diameter/4:
                        break
            else:
                candid_diameter = ann_diameter
            candid_info_list.append(CandidInfoTuple(
                is_nodule, uid, candid_xyz_center, candid_diameter
            ))
    candid_info_list.sort(reverse=True)
    return candid_info_list




str(data_path)



# +
XYZTuple = namedtuple('XYZTuple', ['x', 'y', 'z'])


def irc2xyz(direction_cosine_matrix, irc_coords, spacing_xyz, origin_xyz):
    cri_array = np.array(irc_coords)[::-1]
    spacing_array = np.array(spacing_xyz)
    origin_xyz_array = np.array(origin_xyz)
    xyz_array = direction_cosine_matrix@(cri_array *
                                         spacing_array)+origin_xyz_array
    return XYZTuple(*xyz_array)

IRCTuple = namedtuple('IRCTuple', ['index', 'row', 'column'])


def xyz2irc(xyz_center_coords, origin_xyz, direction_cosine_matrix, spacing_xyz):
    xyz_array = np.array(xyz_center_coords)
    origin_xyz_array = np.array(origin_xyz)
    cri_array = (
        xyz_array - origin_xyz_array)@np.linalg.inv(direction_cosine_matrix)/spacing_xyz
    cri_array = np.round(cri_array)
    return IRCTuple(*cri_array[::-1])



# -

class CT():
    def __init__(self, uid):
        # did not understand why zeroth index?
        mhd_path = glob.glob(os.path.join(
            data_path, "subset*", "{}.mhd".format(uid)))[0]
        mhd_raw_image = sitk.ReadImage(mhd_path)
        ct_array = np.array(sitk.GetArrayFromImage(
            mhd_raw_image), dtype=float)
        ct_array = np.clip(ct_array, -1000, 1000)
        self.uid = uid

        self.hounsfield_array = ct_array
        self.origin_xyz = XYZTuple(*mhd_raw_image.GetOrigin())
        self.spacing = XYZTuple(*mhd_raw_image.GetSpacing())
        self.direction_cosine_matrix = np.array(
            mhd_raw_image.GetDirection()).reshape(3, 3)

    def get_raw_candidate(self, xyz_center_coords, irc_width):
        irc_center_coords = xyz2irc(
            xyz_center_coords,
            self.origin_xyz,
            self.direction_cosine_matrix,
            self.spacing
        )
        slice_list = []
        for axis, center_val in enumerate(irc_center_coords):
            start_ndx = int(round(center_val - irc_width[axis]/2))
            end_ndx = int(round(center_val + irc_width[axis]/2))
            assert center_val >= 0 and center_val < self.hounsfield_array.shape[axis], repr(
                [self.series_uid, xyz_center_coords, self.origin_xyz, self.vxSize_xyz, irc_center_coords, axis])
            if start_ndx < 0:
                log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                    self.series_uid, xyz_center_coords, irc_center_coords, self.hu_a.shape, irc_width))

                start_ndx = 0
                end_ndx = int(irc_width[axis])
            if end_ndx > self.hounsfield_array.shape[axis]:
                log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                    self.series_uid, xyz_center_coords, irc_center_coords, self.hu_a.shape, irc_width))

                end_ndx = self.hounsfield_array.shape[axis]
                start_ndx = int(
                    self.hounsfield_array.shape[axis]-irc_width[axis])

            slice_list.append(slice(start_ndx, end_ndx))
            ct_chunk_array = self.hounsfield_array[tuple(slice_list)]

            return ct_chunk_array, irc_center_coords




@functools.lru_cache(1, typed=True)
def getCT(uid):
    return CT(uid)




# +


@raw_cache.memoize(typed=True)
def get_ct_raw_candidate(uid, xyz_center_coords, irc_width):
    ct = CT(uid)
    ct_chunk_array, irc_center_coords = ct.get_raw_candidate(
        xyz_center_coords, irc_width)
    return ct_chunk_array, irc_center_coords



# -

class LunaDataset(Dataset):
    def __init__(self,
                 data_directory=None,
                 validation_stride=0,
                 is_validation_bool=None,
                 uid=None):
        self.candid_info_list = copy.copy(
            get_candid_info_list(data_directory))
        if uid:
            self.candid_info_list = [
                x for x in self.candid_info_list if x.uid == uid]
        if is_validation_bool:
            assert validation_stride > 0, validation_stride
            self.candid_info_list = self.candid_info_list[::validation_stride]
            assert self.candid_info_list

        # The "elif not is_validation_bool" case is omitted because
        # "is_validation_bool" is a boolean variable that evaluates to False when it is not True.
        # Therefore, Python automatically proceeds to the next statement, which is "elif validation_stride > 0".
        # This branch implicitly covers the scenario where "is_validation_bool" is False or None,
        # effectively handling the same logic as "elif not is_validation_bool".

        elif validation_stride > 0:
            del self.candid_info_list[::validation_stride]
            assert self.candid_info_list

        log.info("{!r}:{} {}".format(self, len(self.candid_info_list),
                                     "Validation Set" if is_validation_bool else "Training Set"))

    def __len__(self):
        return len(self.candid_info_list)

    def __getitem__(self, ndx):
        candid_info_tuple = self.candid_info_list[ndx]
        irc_width = (48, 48, 48)
        cnadid_array, irc_center_coords = get_ct_raw_candidate(candid_info_tuple.uid,
                                                               candid_info_tuple.center_xyz,
                                                               irc_width)
        candid_tensor = torch.from_numpy(
            cnadid_array).to(torch.float32).unsqueeze(0)
        label_tensor = torch.tensor([not candid_info_tuple.is_nodule,
                                    candid_info_tuple.is_nodule], dtype=torch.long)

        return (candid_tensor,
                label_tensor,
                candid_info_tuple.uid,
                torch.tensor(irc_center_coords))




def import_module(module_name: str, submodule_name=None):
    if submodule_name is None and ":" in module_name:
        modeule_name, submodule_name = module_name.split(":")
    module = __import__(module_name)
    for name in module_name.split('.')[1:]:
        module = getattr(module, name)
    if submodule_name:
        try:
            getattr(module, submodule_name)
        except:
            raise ImportError('{}.{}'.format(modeule_name, submodule_name))
    return module




# +
def get_system_specs():
    n_cpu = multiprocessing.cpu_count()
    cpu_info = f"CPUs: {n_cpu}"

    ram = psutil.virtual_memory()
    total_ram = ram.total / (1024 ** 3)
    available_ram = ram.available / (1024 ** 3)
    ram_info = f"Total RAM: {total_ram:.2f} GB,\n\tAvailable RAM: {available_ram:.2f} GB ({100 - ram.percent:.1f}%)"

    n_gpu = len(GPUtil.getGPUs())
    gpu_info = f"GPUs: {n_gpu}" if n_gpu > 0 else "GPUs: None"

    disk = psutil.disk_usage('./')
    total_disk = disk.total / (1024 ** 3)
    available_disk = disk.free / (1024 ** 3)
    disk_info = f"Total disk: {total_disk:.2f} GB,\n\tAvailable disk: {available_disk:.2f} GB ({100 - disk.percent:.1f}%)"

    formatted_output = (
        f"{cpu_info}\n"
        f"{ram_info}\n"
        f"{gpu_info}\n"
        f"{disk_info}"
    )

    return formatted_output


print(get_system_specs().split("\n")[0][-1])


# +

def run(app, *argv):
    argv = list(argv)

    # Dynamically determine the number of workers
    n_cpu = get_system_specs().split("\n")[0][-1]
    argv.insert(0, f'--num-workers={n_cpu}')
    log.info("Running: {}({!r}).main()".format(app, argv))

    app_cls = import_module(*app.rsplit('.', 1))
    app_cls(argv).main()
    log.info("Finished: {}({!r}).main()".format(app, argv))



# +
class LunaTrainingApp():
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--num-workers',
                            help='Number of worker processes for background data loading',
                            default=8,
                            type=int,
                            )
        parser.add_argument('--batch-size',
                            help='Batch size to use for training',
                            default=32,
                            type=int,
                            )
        parser.add_argument('--epochs',
                            help='Number of epochs to train for',
                            default=1,
                            type=int,
                            )

        parser.add_argument('--tb-prefix',
                            default='p2ch11',
                            help="Data prefix to use for TensorBoard run. Defaults to chapter.",
                            )

        parser.add_argument('comment',
                            help="Comment suffix for Tensorboard run.",
                            nargs='?',
                            default='dlwpt',
                            )
        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime(
            "%Y-%m-%d_%H.%M.%S")  # search for ("%Y%m%d-%H%M%S")
        self.training_writer = None
        self.validation_writer = None
        self.total_training_sample_count = 0
        self.cuda_availble = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda_availble else "cpu")
        self.model = self.init_model()
        self.optimizer = init_optimizer()

    def init_model(self):
        model = LunaModel()
        if self.cuda_available:
            log.info("Using CUDA; {} devices.".format(
                torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                # Initialize the distributed environment
                dist.init_process_group(backend='nccl')
                # Get the local rank (GPU id) from the environment
                local_rank = int(os.environ["LOCAL_RANK"])
                torch.cuda.set_device(local_rank)
                model = model.to(self.device)
                model = DistributedDataParallel(
                    model, device_ids=[local_rank])
            else:
                model = model.to(self.device)
        return model

    def init_optimizer(self):
        return Adam(self.model.parameters(), lr=0.001, momentum=0.90)

    def init_training_loader(self):
        training_dataset = LunaDataset(validation_stride=0,
                                       is_validation_bool=None)
        bath_size = self.cli_args.batch_size
        if self.cuda_availble:
            batch_size *= torch.cuda.device_count()
        training_loader = DataLoader(training_dataset,
                                     batch_size=batch_size,
                                     num_workers=self.cli_args.num_workers,
                                     pin_memory=self.cuda_availble)  # what is "pin_memory" for?
        return training_loader

    def init_validation_loader(self):
        validation_dataset = LunaDataset(validation_stride=10,
                                         is_validation_bool=True)
        batch_size = self.cli_args.batch_size
        if self.cuda_availble:
            batch_size *= torch.cuda.device_count()
        validation_loader = DataLoader(validation_dataset,
                                       batch_size=batch_size,
                                       num_workers=self.cli_args.num_workers,
                                       pin_memory=self.cuda_availble)
        return validation_loader
    # the rest of the code to be typed

    def initTensorboardWriters(self):
        if self.training_writer is None:
            log_dir = os.path.join(
                'runs', self.cli_args.tb_prefix, self.time_str)
            self.training_writer = SummaryWriter(
                log_dir=log_dir+'-training_class-'+self.cli_args.comment)
            self.validation_writer = SummaryWriter(
                log_dir=log_dir+'-validation_class-'+self.cli_args.comment
            )

    def main(self):
        log.info(f"Starting {type(self).__name__}, {self.cli_args}")
        training_loader = self.init_training_loader()
        validation_loader = self.init_validation_loader()

        for epoch_ndx in range(1, self.cli_args.epochs + 1):
            log.info(f"Epoch {epoch_ndx} of {self.cli_args.epochs}, "
                     f"{len(training_loader)} train batches, "
                     f"{len(validation_loader)} validation batches, "
                     f"batch size {self.cli_args.batch_size} "
                     f"* {torch.cuda.device_count() if self.cuda_availble else 1} "
                     f"{int((torch.cuda.device_count() if self.cuda_availble else 1)) * int(self.cli_args.batch_size)})")
            train_metrcis_tensor = self.do_training(epoch_ndx, training_loader)
            self.logMetrics(epoch_ndx, 'train', train_metrcis_tensor)

            validation_metrcis_tensor = self.do_validation(
                epoch_ndx, validation_loader)
            self.logMetrics(epoch_ndx, 'validation', validation_metrcis_tensor)

        if hasattr(self, 'training_writer'):
            self.training_writer.close()
            self.validation_writer.close()

    def do_training(self, epoch_ndx, training_loader):
        self.model.train()
        train_metrics_tensorensor= torch.zeros(METRICS_SIZE,
                                       len(training_loader.dataset),
                                       device=self.device)
        batch_iter = enumerateWithEstimate(training_loader,
                                           "Epoch {} training".format(
                                               epoch_ndx),
                                           start_ndx=training_loader.num_workers)
        for batch_ndx, batch_tuple in batch_iter:
            self.optimizer.zero_grad()
            loss_var=self.compute_batch_loss(batch_ndx,
                                             batch_tuple,
                                             training_loader.batch_size,
                                             train_metrics_tensorensor)
            loss_var.backward()
            self.optimizer.step()
            # This is for adding the model graph to TensorBoard.
            if epoch_ndx == 1 and batch_ndx == 0:
                with torch.no_grad():
                    model = LunaModel()
                    self.training_writer.add_graph(
                        model, batch_tuple[0], verbose=True)
                    self.training_writer.close()
            self.total_training_samples_count += len(training_loader.datset)
            return train_metrics_tensorensor.to('cpu')

    def do_validation(self, epoch_ndx, validation_loader):
        with torch.no_grad():
            self.model.eval()
            validation_metrics_tensorensor= torch.zeros(METRICS_SIZE,
                                                len(validation_loader.dataset),
                                                device=self.device)
            batch_iter = enumerateWithEstimate(validation_loader,
                                               "Epoch {} validation".format(
                                                   epoch_ndx),
                                               start_ndx=validation_loader.num_workers)
            for batch_ndx, batch_tuple in batch_iter:
                self.compute_batch_loss(batch_ndx,
                                        batch_tuple,
                                        validation_loader.batch_size,
                                        validation_metrics_tensorensor)
            return validation_metrics_tensorensor.to('cpu')
    # did not understand the following

    def compute_batch_loss(self, batch_ndx, batch_tuple, batch_size, metrics_tensorensor):
        input_tensor, label_tensor, uid_list, irc_center_coords_tensor = batch_tuple
        # what is "non_blocking"?
        input_tensor= input_tensor.to(self.device, non_blocking=True)
        # not candid_info_tuple.is_nodule |
        label_tensor= label_tensor.to(self.device, non_blocking=True)
        # one-hot encoded for negative class

        loss_func = nn.CrossEntropyLoss(reduction='none')
        logits_tensor, probability_tensor= self.model(input_tensor)

        loss_tensor= loss_func(logits_tensor, label_tensor[:, 1])
        start_ndx = batch_ndx*batch_size
        end_ndx = start_ndx+label_tensor.size(0)

        metrics_tensorensor[METRICS_LABEL_NDX,
                   start_ndx:end_ndx] = label_tensor[:, 1].detach()
        metrics_tensorensor[METRICS_PRED_NDX,
                   start_ndx:end_ndx] = probability_tensor[:, 1].detach()
        metrics_tensorensor[METRICS_LOSS_NDX, start_ndx:end_ndx] = loss_tensor.detach()

        return torch.tensor(loss_tensor.mean(), dtype=torch.float)

    def logMetrics(
        self,
        epoch_ndx,
        mode_str,
        metrics_tensor,
        classificationThreshold=0.5,
    ):

        self.initTensorboardWriters()
        log.info("E{} {}".format(
            epoch_ndx,
            type(self).__name__,
        ))

        negative_label_mask = metrics_tensor[METRICS_LABEL_NDX] <= classificationThreshold
        negative_prediction_mask = metrics_tensor[METRICS_PRED_NDX] <= classificationThreshold

        positive_label_mask = ~negative_label_mask
        positive_prediction_mask = ~negative_prediction_mask

        n_negative = int(negative_label_mask.sum())
        n_positive = int(positive_label_mask.sum())

        negative_correct = int((negative_label_mask & negative_prediction_mask).sum())
        positive_correct = int((positive_label_mask & positive_prediction_mask).sum())

        metrics_dict = {}
        metrics_dict['loss/all'] = \
            metrics_tensor[METRICS_LOSS_NDX].mean()
        metrics_dict['loss/neg'] = \
            metrics_tensor[METRICS_LOSS_NDX, negative_label_mask].mean()
        metrics_dict['loss/pos'] = \
            metrics_tensor[METRICS_LOSS_NDX, positive_label_mask].mean()

        metrics_dict['correct/all'] = (positive_correct + negative_correct) \
            / np.float32(metrics_tensor.shape[1]) * 100
        metrics_dict['correct/neg'] = negative_correct / np.float32(n_negative) * 100
        metrics_dict['correct/pos'] = positive_correct / np.float32(n_positive) * 100

        log.info(
            ("E{} {:8} {loss/all:.4f} loss, "
                + "{correct/all:-5.1f}% correct, "
             ).format(
                epoch_ndx,
                mode_str,
                **metrics_dict,
            )
        )
        log.info(
            ("E{} {:8} {loss/neg:.4f} loss, "
                + "{correct/neg:-5.1f}% correct ({negative_correct:} of {n_negative:})"
             ).format(
                epoch_ndx,
                mode_str + '_neg',
                negative_correct=negative_correct,
                n_negative=n_negative,
                **metrics_dict,
            )
        )
        log.info(
            ("E{} {:8} {loss/pos:.4f} loss, "
                + "{correct/pos:-5.1f}% correct ({positive_correct:} of {n_positive:})"
             ).format(
                epoch_ndx,
                mode_str + '_pos',
                positive_correct=positive_correct,
                n_positive=n_positive,
                **metrics_dict,
            )
        )

        writer = getattr(self, mode_str + '_writer')

        for key, value in metrics_dict.items():
            writer.add_scalar(key, value, self.totalTrainingSamples_count)

        writer.add_pr_curve(
            'pr',
            metrics_tensor[METRICS_LABEL_NDX],
            metrics_tensor[METRICS_PRED_NDX],
            self.totalTrainingSamples_count,
        )

        bins = [x/50.0 for x in range(51)]

        negHist_mask = negative_label_mask & (metrics_tensor[METRICS_PRED_NDX] > 0.01)
        posHist_mask = positive_label_mask & (metrics_tensor[METRICS_PRED_NDX] < 0.99)

        if negHist_mask.any():
            writer.add_histogram(
                'is_neg',
                metrics_tensor[METRICS_PRED_NDX, negHist_mask],
                self.totalTrainingSamples_count,
                bins=bins,
            )
        if posHist_mask.any():
            writer.add_histogram(
                'is_pos',
                metrics_tensor[METRICS_PRED_NDX, posHist_mask],
                self.totalTrainingSamples_count,
                bins=bins,
            )

        # score = 1 \
        #     + metrics_dict['pr/f1_score'] \
        #     - metrics_dict['loss/mal'] * 0.01 \
        #     - metrics_dict['loss/all'] * 0.0001
        #
        # return score

    def logMetrics(
        self,
        epoch_ndx: int,
        mode_str: str,
        metrics_tensor: torch.Tensor,
        classificationThreshold: float = 0.5,
    ) -> None:
        """
        Computes and logs metrics for the given epoch.

        Args:
        - epoch_ndx: The index of the epoch.
        - mode_str: The mode of the metrics ('trn' or 'val').
        - metrics_tensor: A tensor containing the metrics.
        - classificationThreshold: The classification threshold.
        """
        # Initialize Tensorboard writers
        self.initTensorboardWriters()

        # Logging information about the epoch
        log.info(f"E{epoch_ndx} {type(self).__name__}")

        # Masks to filter positive and negative labels and predictions
        negative_label_mask: torch.BoolTensor = metrics_tensor[METRICS_LABEL_NDX] <= classificationThreshold
        negative_prediction_mask: torch.BoolTensor = metrics_tensor[METRICS_PRED_NDX] <= classificationThreshold
        positive_label_mask: torch.BoolTensor = ~negative_label_mask
        positive_prediction_mask: torch.BoolTensor = ~negative_prediction_mask

        # Counting positive and negative samples
        n_negative: int = int(negative_label_mask.sum())
        n_positive: int = int(positive_label_mask.sum())

        # Counting correct predictions
        negative_correct: int = int((negative_label_mask & negative_prediction_mask).sum())
        positive_correct: int = int((positive_label_mask & positive_prediction_mask).sum())

        # Dictionary to store metrics
        metrics_dict: Dict[str, Union[float, int]] = {}

        # Loss metrics
        metrics_dict['loss/all'] = metrics_tensor[METRICS_LOSS_NDX].mean().item()
        metrics_dict['loss/neg'] = metrics_tensor[METRICS_LOSS_NDX,
                                             negative_label_mask].mean().item()
        metrics_dict['loss/pos'] = metrics_tensor[METRICS_LOSS_NDX,
                                             positive_label_mask].mean().item()

        # Correct prediction metrics
        metrics_dict['correct/all'] = (positive_correct + negative_correct) / \
            np.float32(metrics_tensor.shape[1]) * 100
        metrics_dict['correct/neg'] = negative_correct / np.float32(n_negative) * 100
        metrics_dict['correct/pos'] = positive_correct / np.float32(n_positive) * 100

        # Logging loss and correct prediction metrics
        log.info(
            f"E{epoch_ndx} {mode_str}:8> loss: {metrics_dict['loss/all']:.4f}, "
            f"correct: {metrics_dict['correct/all']:-5.1f}% "
            f"({positive_correct + negative_correct:}/{metrics_tensor.shape[1]:})"
        )
        log.info(
            f"E{epoch_ndx} {mode_str}_neg:8> loss: {metrics_dict['loss/neg']:.4f}, "
            f"correct: {metrics_dict['correct/neg']:-5.1f}% ({negative_correct:}/{n_negative:})"
        )
        log.info(
            f"E{epoch_ndx} {mode_str}_pos:8> loss: {metrics_dict['loss/pos']:.4f}, "
            f"correct: {metrics_dict['correct/pos']:-5.1f}% ({positive_correct:}/{n_positive:})"
        )
        # Get the appropriate writer for logging
        writer = getattr(self, mode_str + '_writer')

        # Log scalar metrics
        for key, value in metrics_dict.items():
            writer.add_scalar(key, value, self.totalTrainingSamples_count)

        # Log precision-recall curve
        writer.add_pr_curve(
            'pr',
            metrics_tensor[METRICS_LABEL_NDX],
            metrics_tensor[METRICS_PRED_NDX],
            self.totalTrainingSamples_count,
        )

        # Bins for histogram
        bins = [x/50.0 for x in range(51)]

        # Masks for histogram
        negHist_mask: torch.BoolTensor = negative_label_mask & (
            metrics_tensor[METRICS_PRED_NDX] > 0.01)
        posHist_mask: torch.BoolTensor = positive_label_mask & (
            metrics_tensor[METRICS_PRED_NDX] < 0.99)

        # Log histograms
        if negHist_mask.any():
            writer.add_histogram(
                'is_neg',
                metrics_tensor[METRICS_PRED_NDX, negHist_mask],
                self.totalTrainingSamples_count,
                bins=bins,
            )
        if posHist_mask.any():
            writer.add_histogram(
                'is_pos',
                metrics_tensor[METRICS_PRED_NDX, posHist_mask],
                self.totalTrainingSamples_count,
                bins=bins,
            )


# if __name__ == '__main__':
#     LunaTrainingApp().main()

# -


class LunaBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.conv1 = nn.Conv3d(in_channels, out_channels,
                               kernel_size=3, padding=1, bias=True)
        self.relu1 = nn.ReLU(insplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels,
                               kernel_size=3, padding=1, bias=True)
        self.relu2 = nn.ReLU(insplace=True)

        self.maxpool = nn.MaxPool3d(2, 2)

        def forward(self, input_batch):
            block_out = self.relu1(self.conv1(input_batch))
            block_out = self.relu2(self.conv2(block_out))

            return self.maxpool(block_out)



# +


class LunaModel(nn.Module):
    def __inti__(self, in_channels=1, out_channels=8):
        super.__init__()
        # tail
        self.tail_batchnorm = nn.BatchNorm3d(1)
        # body
        self.block1 = LunaBlock(in_channels, out_channels)
        self.block2 = LunaBlock(out_channels, out_channels*2)
        self.block3 = LunaBlock(out_channels*2, out_channels*4)
        self.block4 = LunaBlock(out_channels*4, out_channels*8)
        # head
        self.head_linear = nn.Linear(out_channels*8, 2)
        self.head_softmax = nn.Softmax(dim=1)

        self._init_weighta()

    def _init_wights(self):
        for module in self.modules():
            if type(module) in {
                nn.Linear,
                nn.Conv3d,
                nn.Conv2d,
                nn.ConvTranspose2d,
                nn.ConvTranspose3d,
            }:
                nn.init.kaiming_normal_(
                    module.weight.data,
                    a=0,
                    mode='fan_out',
                    nonlinearity='relu',
                )
                if module.bais is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(
                        module.weight.data
                    )
                    bound = 1 / math.sqrt(fan_out)
                    # the mean is -bound and the standard deviation is bound,
                    # which effectively sets the bias values within
                    # a symmetric range around zero, controlled by the previously calculated bound.
                    nn.init.normal_(module.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.tail_batchnorm(input_batch)

        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        block_out = self.block4(block_out)
        flat_conv = block_out.view(block_out.size[0], -1)
        linear_output = self.head_linear(flat_conv)
        return linear_output, self.head_softmax(linear_output)




# +
def importstr(module, submodule=None):
    if submodule is None and ':' in module:
        module,submodule = module.split(':')
        module=__import__(module)
    for module_str in module.split('.')[1:]:
        module=getattr(module, module_str)
    if sub_module:
        try:
            return getattr(module, submodule)
        except:
            raise ImportError('{}.{}'.format(module, submodule))
    return module
        

# -

def run(app, *argv):
    argv=list(argv)
    n_cpu=get_system_specs().split("\n")[0][-1]
    argv.inset(0, f'--num-workers={n_cpu}')
    log.info("Running: {}({!r}).main()".format(app, argv))
    
    app_cls=importstr(*app.rsplit('.',1))
    app_cls(argv).main()
    log.info("Finished: {}.{!r}).main()".format(app, argv))

