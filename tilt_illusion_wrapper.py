import time
import sys
import numpy as np
import os
import tilt_illusion

class Args:
    def __init__(self):
        self.save_images = True
        self.save_metadata = True

t = time.time()
args = Args()
## Constraints
num_machines = int(sys.argv[1])
args.batch_id = int(sys.argv[2])
total_images = int(sys.argv[3])
args.n_images = total_images/num_machines

if len(sys.argv)==4:
    print('Using default path...')
    dataset_root = '/Users/junkyungkim/Desktop/tilt_illusion'
elif len(sys.argv)==5:
    print('Using custom save path...')
    dataset_root = str(sys.argv[4])
else:
    raise ValueError('wrong number of args')

## Parameters
args.image_size = [500, 500]
args.r1_range = [80, 240]
args.lambda_range = [30, 90]

################################# train
dataset_subpath = 'train'
args.dataset_path = os.path.join(dataset_root, dataset_subpath)
tilt_illusion.from_wrapper(args, train=True)

################################# test
dataset_subpath = 'test'
args.r1_range = [args.r1_range[0]/2, args.r1_range[1]/2]
args.dataset_path = os.path.join(dataset_root, dataset_subpath)
tilt_illusion.from_wrapper(args, train=False)
