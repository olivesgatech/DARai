# read csv files in /mnt/HD_18T_pt1/yavuz/data/IMU_LeftArm

import os
import numpy as np
import pandas as pd
from glob import glob
import scipy.io
import argparse

modality='IMU_LeftArm'
parser = argparse.ArgumentParser()

parser.add_argument('--data_path', type=str, required=True, 
                    help='Path to the data directory')
args = parser.parse_args()
data_path = args.data_path
save_path = f'./darai/inertial/'

if not os.path.exists(save_path):
    os.makedirs(save_path)

#sort folders
act=0

folders=['Carrying object',
'Playing video game',
'Reading',
'Sleeping',
'Exercising',
'Using handheld smart devices',
'Watching TV',
'Working on a computer',
 'Writing',
 'Cleaning dishes',
'Dining',
'Stocking up pantry',
'Making a cup of coffee in coffee maker',
'Making a cup of instant coffee',
'Making a salad',
'Making pancake'
 ]
folders.sort()


for folder in folders:
    act=act+1
    files=glob(data_path+folder+"/*.csv")
    print(files)
    for file in files:
        data=np.loadtxt(file, delimiter=",", dtype=float,skiprows=1,usecols=range(1,10)).T
        N=data.shape[1]
        file_name=file.split('/')[-1][:-4] #remove .csv
        subject_name=int(file_name[:-2])
        session=int(file_name[3])
        trial=session
        data=data.T
        new_file_name=f"a{act}_s{subject_name}_t{trial}_inertial.mat"
        
        scipy.io.savemat(save_path+new_file_name, {'d_iner':data})
        

modality='IMU_RightArm'

act=0
for folder in folders:
    act=act+1
    files=glob(data_path+folder+"/*.csv")
    for file in files:
        data=np.loadtxt(file, delimiter=",", dtype=float,skiprows=1,usecols=range(1,10)).T
        N=data.shape[1]
        file_name=file.split('/')[-1][:-4] #remove .csv
        subject_name=int(file_name[:-2])
        session=int(file_name[3])
        trial=session+4
        data=data.T
        new_file_name=f"a{act}_s{subject_name}_t{trial}_inertial.mat"
        
        scipy.io.savemat(save_path+new_file_name, {'d_iner':data})
