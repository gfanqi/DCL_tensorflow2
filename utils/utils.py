import os
import pandas as pd
# import torch

from transforms import transforms


# def load_data_transformers(resize_reso=512, crop_reso=448, swap_num=None):
#     if swap_num is None:
#         swap_num = [7, 7]
#     center_resize = 600
#
#     # Normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#
#     data_transforms = {
#             'swap': transforms.Compose([
#                 transforms.Randomswap((swap_num[0], swap_num[1])),
#             ]),
#             'common_aug': transforms.Compose([
#                 transforms.Resize((resize_reso, resize_reso)),
#                 transforms.RandomRotation(degrees=15),
#                 transforms.RandomCrop((crop_reso, crop_reso)),
#                 transforms.RandomHorizontalFlip(),
#             ]),
#             'train_totensor': transforms.Compose([
#                 transforms.Resize((crop_reso, crop_reso)),
#                 # ImageNetPolicy(),
#                 # transforms.ToTensor(),
#                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#             ]),
#             'val_totensor': transforms.Compose([
#                 transforms.Resize((crop_reso, crop_reso)),
#                 # transforms.ToTensor(),
#                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#             ]),
#             'test_totensor': transforms.Compose([
#                 transforms.Resize((resize_reso, resize_reso)),
#                 transforms.CenterCrop((crop_reso, crop_reso)),
#                 # transforms.ToTensor(),
#                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#             ]),
#             'None': None,
#         }
#     return data_transforms

# train_set = dataset(Config=Config,
#                     anno=Config.train_anno,
#                     common_aug=transformers["common_aug"],
#                     swap=transformers["swap"],
#                     totensor=transformers["train_totensor"],
#                     train=True)


import os,time,datetime
import numpy as np
from math import ceil
import datetime

import pdb

def dt():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")

