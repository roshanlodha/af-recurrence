# OS 
import os 
import sys
import pickle
from pathlib import Path

# Math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import random

# .nii
import nibabel as nib
import nilearn as nil

def load_images_df():
    # load samples
    images_df = pd.read_csv("/home/lodhar/af-recurrence/data/vanderbilt_ct_phenotype_2-14-23.csv")
    images_df['nii_dir'] = '/home/lodhar/nii/vandy/' + images_df['study_id'].astype(str) + '.nii.gz'
    images_df['exists'] = images_df.apply(lambda row: os.path.isfile(row['nii_dir']), axis = 1)
    images_df_filtered = images_df[images_df['exists']].drop(['exists'], axis = 1)
    images_df_filtered_shuffled = images_df_filtered.sample(frac=1).reset_index(drop=True)
    return images_df_filtered_shuffled

def get_scan_paths(images_df):
    abnormal_scan_paths = images_df[images_df['recurrence'] == 1]['nii_dir'].values
    print("number of abnormal scans: " + str(len(images_df)))

    normal_scan_paths = images_df[images_df['recurrence'] == 0]['nii_dir'].values
    print("number of normal scans: " + str(len(normal_scan_paths)))
    return abnormal_scan_paths, normal_scan_paths

def read_nifti_file(filepath):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
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
    # Set the desired depth
    desired_depth = 128
    desired_width = 256
    desired_height = 256
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    img = ndi.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndi.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img

def process_scan(path):
    """Read and resize volume"""
    try:
        # Read scan
        volume = read_nifti_file(path)
        # Normalize
        volume = normalize(volume)
        # Resize width, height and depth
        volume = resize_volume(volume)
        if volume is None:
            print(path + " volume is None")
            raise Exception()
        else:
            print(path + " read successfully")
            return volume
    except:
        print(path + " could not be read")

def read_scan_from_path(paths):
    scans = np.array([process_scan(path) for path in paths], dtype = "object") 
    non_filtered_paths = [path for path, scan in zip(paths, scans) if scan is not None]
    
    print("filtering scans...")
    scans = np.array(list(filter(lambda item: item is not None, scans)))

    return scans, non_filtered_paths

def split_scans(split, normal_scans, abnormal_scans, normal_scan_paths, abnormal_scan_paths):
    n_scans = min(len(abnormal_scan_paths), len(normal_scan_paths))
    split_point = int(split * n_scans)

    normal_labels = np.array([0 for _ in range(len(normal_scans))])
    abnormal_labels = np.array([1 for _ in range(len(abnormal_scans))])

    x_train = np.concatenate((abnormal_scans[:split_point], normal_scans[:split_point]), axis=0)
    y_train = np.concatenate((abnormal_labels[:split_point], normal_labels[:split_point]), axis=0)
    train_paths = np.concatenate((abnormal_scan_paths[:split_point], normal_scan_paths[:split_point]), axis=0)

    x_val = np.concatenate((abnormal_scans[split_point:n_scans], normal_scans[split_point:n_scans]), axis=0)
    y_val = np.concatenate((abnormal_labels[split_point:n_scans], normal_labels[split_point:n_scans]), axis=0)
    val_paths = np.concatenate((abnormal_scan_paths[split_point:n_scans], normal_scan_paths[split_point:n_scans]), axis=0)

    return x_train, y_train, x_val, y_val, train_paths, val_paths

def main():
    images_df = load_images_df()
    abnormal_scan_paths, normal_scan_paths = get_scan_paths(images_df)
    
    print("reading normal scans...")
    normal_scans, normal_scan_paths = read_scan_from_path(normal_scan_paths)

    print("reading abnormal scans...")
    abnormal_scans, abnormal_scan_paths = read_scan_from_path(abnormal_scan_paths)

    print("performing test-train split on scans...")
    x_train, y_train, x_val, y_val, train_paths, val_paths = split_scans(0.7, normal_scans, abnormal_scans, normal_scan_paths, abnormal_scan_paths)

    print(
        "Number of samples in train and validation are %d and %d."
        % (x_train.shape[0], x_val.shape[0])
    )

    images_df['is_train'] = images_df['nii_dir'].apply(lambda x: 1 if x in train_paths else 0)
    images_df['is_val'] = images_df['nii_dir'].apply(lambda x: 1 if x in val_paths else 0)
    images_df.to_csv('/home/lodhar/af-recurrence/data/images_df_filtered_shuffled_downsamples.csv')

    dataset_dict = {"x_train": x_train, "x_val": x_val, "y_train": y_train, "y_val": y_val}
    with open('/home/lodhar/af-recurrence/data/train_val_256_downsampled.pkl', 'wb') as file:
        pickle.dump(dataset_dict, file)

if __name__ == '__main__':
    main()
