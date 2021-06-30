#!/usr/bin/python3
import os
import glob
import sys
import random
import numpy as np
import pickle
from numpy.lib.format import open_memmap
from tqdm import tqdm

import argparse

def gendata(data_path, out_path, ignored_sample_path=None, benchmark='cs', part='train'):
    print("Running {}, {}".format(benchmark, part))
    training_subjects = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
    training_cameras = [2, 3]
    max_body = 2
    num_joint = 17
    max_frame = 300
    
    ignored_samples = []
    sample_name = []
    sample_label = []
    for filename in os.listdir(data_path):
        if filename in ignored_samples:
            continue
        action_class = int(
            filename[filename.find('A') + 1:filename.find('A') + 4])
        subject_id = int(
            filename[filename.find('P') + 1:filename.find('P') + 4])
        camera_id = int(
            filename[filename.find('C') + 1:filename.find('C') + 4])

        if benchmark == 'cv':
            istraining = (camera_id in training_cameras)
        elif benchmark == 'cs':
            istraining = (subject_id in training_subjects)
        else:
            raise ValueError()

        if part == 'train':
            issample = istraining
        elif part == 'test':
            issample = not (istraining)
        else:
            raise ValueError()

        if issample:
            sample_name.append(filename)
            sample_label.append(action_class - 1)

    if part == 'train':
        number_of_train_samples = int(len(sample_name)*0.8)
        train_list = random.sample(range(0,len(sample_name)),number_of_train_samples)
        val_list = [i for i in range(0,len(sample_name)) if i not in train_list]
        sample_name_train = [sample_name[i] for i in train_list]
        sample_label_train = [sample_label[i] for i in train_list]
        sample_name_val = [sample_name[i] for i in val_list]
        sample_label_val = [sample_label[i] for i in val_list]

        with open('{}/{}_label.pkl'.format(out_path, 'train'), 'wb') as f:
            pickle.dump((sample_name_train, list(sample_label_train)), f)
        fp_train = open_memmap(
            '{}/{}_data.npy'.format(out_path, 'train'),
            dtype='float32',
            mode='w+',
            shape=(len(sample_label_train), 3, max_frame, num_joint, max_body))

        for i in tqdm(range(len(sample_name_train))):
            s = sample_name_train[i]
            with open(os.path.join(data_path, s), 'rb') as f:
                #frames contains (frame, body, joint, xyz)
                #should be (xyz, frame, joint, body)
                #ie transpose(3, 0, 2, 1)
                data = np.load(f).transpose(3, 0, 2, 1)
                fp_train[i, :, 0:data.shape[1], :, :] = data

        with open('{}/{}_label.pkl'.format(out_path, 'val'), 'wb') as f:
            pickle.dump((sample_name_val, list(sample_label_val)), f)
        fp_val = open_memmap(
            '{}/{}_data.npy'.format(out_path, 'val'),
            dtype='float32',
            mode='w+',
            shape=(len(sample_label_val), 3, max_frame, num_joint, max_body))

        for i in tqdm(range(len(sample_name_val))):
            s = sample_name_val[i]
            with open(os.path.join(data_path, s), 'rb') as f:
                #frames contains (frame, body, joint, xyz)
                #should be (xyz, frame, joint, body)
                #ie transpose(3, 0, 2, 1)
                data = np.load(f).transpose(3, 0, 2, 1)
                fp_val[i, :, 0:data.shape[1], :, :] = data

    else:
        with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
            pickle.dump((sample_name, list(sample_label)), f)

        fp = open_memmap(
            '{}/{}_data.npy'.format(out_path, part),
            dtype='float32',
            mode='w+',
            shape=(len(sample_label), 3, max_frame, num_joint, max_body))

        for i in tqdm(range(len(sample_name))):
            s = sample_name[i]
            with open(os.path.join(data_path, s), 'rb') as f:
                #frames contains (frame, body, joint, xyz)
                #should be (xyz, frame, joint, body)
                #ie transpose(3, 0, 2, 1)
                data = np.load(f).transpose(3, 0, 2, 1)
                fp[i, :, 0:data.shape[1], :, :] = data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NTURGB+D Data Converter.')
    parser.add_argument(
        '--data_path', default='/home/magnus/VSCODIUM_projects/camera_parameters/data/generated_data/')
    parser.add_argument(
        '--ignored_sample_path',
        default='/home/magnus/VSCODIUM_projects/camera_parameters/data/samples_with_missing_skeletons.txt')
    parser.add_argument('--out_folder', default='/home/magnus/VSCODIUM_projects/camera_parameters/data/')
    benchmark = ['cs', 'cv']
    part = ['train', 'test']
    for b in benchmark:
        for p in part:
            out_path = os.path.join(out_folder, b)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            gendata(data_path, out_path, benchmark=b, part=p)
