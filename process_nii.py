#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage
from sklearn.model_selection import train_test_split
import pickle as pkl
from keras import layers
import logging

def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def find_nifti_dim(filepath):
    try:
        scan = nib.load(filepath)
        dim = scan.header.get_data_shape()
        if len(dim) >= 3:
            return dim[:3]
        else:
            return (None, None, None)
    except Exception as e:
        logging.error(f"Error reading {filepath}: {e}")
        return (None, None, None)

def read_nifti_file(filepath):
    """Read and load volume"""
    scan = nib.load(filepath)
    scan = scan.get_fdata()
    return scan

def normalize(volume):
    """Normalize the volume"""
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume

def resize_volume(img):
    """Resize across z-axis"""
    desired_depth = 128
    desired_width = 256
    desired_height = 256
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    depth_factor = current_depth / desired_depth
    width_factor = current_width / desired_width
    height_factor = current_height / desired_height
    depth_factor = 1 / depth_factor
    width_factor = 1 / width_factor
    height_factor = 1 / height_factor
    img = ndimage.rotate(img, 90, reshape=False)
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img

def process_scan(path):
    """Read and resize volume"""
    try:
        volume = read_nifti_file(path)
        volume = normalize(volume)
        volume = resize_volume(volume)
        if volume is None:
            logging.error(f"{path} volume is None")
            raise Exception()
        else:
            logging.info(f"{path} read successfully")
            return volume
    except Exception as e:
        logging.error(f"{path} could not be read: {e}")

def load_and_preprocess_images(images_df):
    abnormal_scan_paths = images_df[images_df['recurrence'] == 1]['scan_dir'].values
    normal_scan_paths = images_df[images_df['recurrence'] == 0]['scan_dir'].values

    logging.info("Reading abnormal scans...")
    abnormal_scans = np.array([process_scan(path) for path in abnormal_scan_paths], dtype="object")
    non_filtered_abnormal_paths = [path for path, scan in zip(abnormal_scan_paths, abnormal_scans) if scan is not None]
    logging.info("Filtering abnormal scans...")
    abnormal_scans = np.array(list(filter(lambda item: item is not None, abnormal_scans)))

    logging.info("Reading normal scans...")
    normal_scans = np.array([process_scan(path) for path in normal_scan_paths], dtype="object")
    non_filtered_normal_paths = [path for path, scan in zip(normal_scan_paths, normal_scans) if scan is not None]
    logging.info("Filtering normal scans...")
    normal_scans = np.array(list(filter(lambda item: item is not None, normal_scans)))

    return abnormal_scans, normal_scans, non_filtered_abnormal_paths, non_filtered_normal_paths

def split_and_upsample(abnormal_scans, normal_scans, abnormal_labels, normal_labels):
    x_data = np.concatenate((abnormal_scans, normal_scans), axis=0)
    y_data = np.concatenate((abnormal_labels, normal_labels), axis=0)

    x_train, x_val, y_train, y_val = train_test_split(
        x_data, y_data, test_size=0.3, random_state=44106, stratify=y_data
    )

    # Perform upsampling on the training set
    train_df = pd.DataFrame({'scan': list(x_train), 'label': y_train})
    no_recur_train = train_df[train_df['label'] == 0]
    recur_train = train_df[train_df['label'] == 1]

    n_samples_needed = len(no_recur_train)
    n_repeats = n_samples_needed // len(recur_train)

    recur_train_repeated = pd.concat([recur_train] * n_repeats, ignore_index=True)
    n_additional_samples = n_samples_needed - len(recur_train_repeated)
    recur_train_additional = recur_train.sample(n=n_additional_samples, replace=True, random_state=44106)

    recur_train_upsampled = pd.concat([recur_train_repeated, recur_train_additional], ignore_index=True)
    train_df_upsampled = pd.concat([no_recur_train, recur_train_upsampled])

    x_train = np.array(train_df_upsampled['scan'].tolist())
    y_train = np.array(train_df_upsampled['label'].tolist())

    return x_train, x_val, y_train, y_val

def save_processed_data(x_train, x_val, y_train, y_val, images_df_filtered, train_paths, val_paths):
    logging.info(f"Number of samples in train and validation are {x_train.shape[0]} and {x_val.shape[0]}.")

    images_df_filtered_upsampled = images_df_filtered.copy()
    images_df_filtered_upsampled['is_training'] = images_df_filtered_upsampled['scan_dir'].apply(lambda x: 1 if x in train_paths else 0)
    images_df_filtered_upsampled['is_validation'] = images_df_filtered_upsampled['scan_dir'].apply(lambda x: 1 if x in val_paths else 0)

    images_df_filtered_upsampled.to_csv("/home/lodhar/af-recurrence/images_df_filtered_upsampled.csv")

    dataset_dict = {"x_train": x_train, "x_val": x_val, "y_train": y_train, "y_val": y_val}

    with open('/home/lodhar/af-recurrence/train_val_data.pkl', 'wb') as file:
        pkl.dump(dataset_dict, file)

def main():
    log_file = "/home/lodhar/af-recurrence/logs/process_nii.log"
    setup_logging(log_file)
    
    images_df = pd.read_csv("/home/lodhar/af-recurrence/input/vanderbilt_ct_phenotype_2-14-23.csv")
    images_df['scan_dir'] = '/home/lodhar/afib-dl/nifti/vandy/' + images_df['study_id'].astype(str) + '.nii.gz'

    images_df[['scan_dim_x', 'scan_dim_y', 'scan_dim_z']] = images_df.apply(
        lambda row: pd.Series(find_nifti_dim(row['scan_dir'])), axis=1
    )
    images_df_filtered = images_df[(images_df['scan_dim_x'] == 512) & (images_df['scan_dim_y'] == 512)]

    abnormal_scans, normal_scans, non_filtered_abnormal_paths, non_filtered_normal_paths = load_and_preprocess_images(images_df_filtered)

    abnormal_labels = np.array([1 for _ in range(len(abnormal_scans))])
    normal_labels = np.array([0 for _ in range(len(normal_scans))])

    x_train, x_val, y_train, y_val = split_and_upsample(abnormal_scans, normal_scans, abnormal_labels, normal_labels)

    train_paths = np.concatenate((non_filtered_abnormal_paths[:len(x_train)//2], non_filtered_normal_paths[:len(x_train)//2]))
    val_paths = np.concatenate((non_filtered_abnormal_paths[len(x_train)//2:], non_filtered_normal_paths[len(x_train)//2:]))

    save_processed_data(x_train, x_val, y_train, y_val, images_df_filtered, train_paths, val_paths)

if __name__ == "__main__":
    # nohup python process_nii.py &> logs/process_nii.txt & 
    main()
