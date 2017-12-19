
import os
os.chdir("Directory where utils.py , vgg16.py , vgg16bn.py and data are stored" )
from __future__ import division,print_function
import json
from glob import glob
import numpy as np
np.set_printoptions(precision=4, linewidth=100)
from matplotlib import pyplot as plt


from utils import plots
from vgg16 import Vgg16


#Path where the data resides.
path = os.getcwd()+"\\KaggleData\\"

batch_size=64
vgg = Vgg16()
batches = vgg.get_batches(path+'train', batch_size=batch_size)
val_batches = vgg.get_batches(path+'valid', batch_size=batch_size*2)
vgg.finetune(batches)
vgg.fit(batches, val_batches, nb_epoch=1)

