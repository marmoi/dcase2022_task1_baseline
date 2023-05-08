import os
import numpy as np
import argparse
import h5py
import time
import pandas as pd
from utils import float32_to_int16
import config

def to_one_hot(k, classes_num):
    target = np.zeros(classes_num)
    target[k] = 1
    return target


def pad_truncate_sequence(x, max_len):
    if len(x) < max_len:
        return np.concatenate((x, np.zeros(max_len - len(x))))
    else:
        return x[0: max_len]


def pack_audio_files_to_hdf5(args):
    # Arguments & parameters
    dataset_file = args.dataset_file.strip("'")
    output_dir = args.workspace.strip("'")

    clip_samples = config.clip_samples
    data_path = config.data_path    # full path of the extracted features
    mel_bins = config.mel_bins

    # Paths - select data
    audios_dir = pd.read_csv(dataset_file, sep='\t', index_col=False)
    audio_names = audios_dir['filename']
    audio_labels = audios_dir['scene_label']
    identifier = audios_dir['identifier']
    source_label = audios_dir['source_label']

    # Path of the output file
    packed_hdf5_path = os.path.join(output_dir,'features.h5')

    meta_dict = {
        'filename': np.array(audio_names),
        'scene_label': np.array(audio_labels),
        'audio_path': data_path,
        'identifier': np.array(identifier),
        'source_label': np.array(source_label)
    }

    audios_num = len(meta_dict['filename'])

    feature_time = time.time()
    with h5py.File(packed_hdf5_path, 'w') as hf:

        hf.create_dataset(
            name='filename',
            shape=(audios_num, ),
            dtype='S50') #The names are pretty long

        hf.create_dataset(
            name='scene_label',
            shape=(audios_num, ),
            dtype='S20')

        hf.create_dataset(
            name='identifier',
            shape=(audios_num, ),
            dtype='S20')

        hf.create_dataset(
            name='source_label',
            shape=(audios_num, ),
            dtype='S10')

        hf.create_dataset(
            name='features',
            shape=(audios_num, mel_bins, clip_samples),
            dtype=np.int16)

        for n in range(audios_num):
            audio_name = meta_dict['filename'][n].split("/")[1][:-3]+'cpickle'
            pickle_file = pd.read_pickle(os.path.join(meta_dict['audio_path'], audio_name))
            features = pickle_file['_data']
            features = float32_to_int16(features)

            hf['filename'][n] = meta_dict['filename'][n].split("/")[1].encode()
            hf['scene_label'][n]= meta_dict['scene_label'][n].encode()
            hf['identifier'][n] = meta_dict['identifier'][n].encode()
            hf['source_label'][n] = meta_dict['source_label'][n].encode()
            hf['features'][n] = features

    print('Write hdf5 to {}'.format(packed_hdf5_path))
    print('Time: {:.3f} s'.format(time.time() - feature_time))



def pack_audio_eval_files_to_hdf5(args):
    # Arguments & parameters
    dataset_file = args.dataset_file.strip("'")
    output_dir = args.workspace.strip("'")

    clip_samples = config.clip_samples
    data_path = config.data_path    # full path of the extracted features
    mel_bins = config.mel_bins

    # Paths - select data
    audios_dir = pd.read_csv(dataset_file, sep='\t', index_col=False)
    audio_names = audios_dir['filename']


    # Path of the output file
    packed_hdf5_path = os.path.join(output_dir,'features_eval.h5')

    meta_dict = {
        'filename': np.array(audio_names),
        'audio_path': data_path,
    }

    audios_num = len(meta_dict['filename'])

    feature_time = time.time()
    with h5py.File(packed_hdf5_path, 'w') as hf:

        hf.create_dataset(
            name='filename',
            shape=(audios_num, ),
            dtype='S50') #The names are pretty long

        hf.create_dataset(
            name='features',
            shape=(audios_num, mel_bins, clip_samples),
            dtype=np.int16)

        for n in range(audios_num):
            audio_name = meta_dict['filename'][n].split("/")[1][:-3]+'cpickle'
            pickle_file = pd.read_pickle(os.path.join(meta_dict['audio_path'], audio_name))
            features = pickle_file['_data']
            features = float32_to_int16(features)

            hf['filename'][n] = meta_dict['filename'][n].split("/")[1].encode()
            hf['features'][n] = features

    print('Write hdf5 to {}'.format(packed_hdf5_path))
    print('Time: {:.3f} s'.format(time.time() - feature_time))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')

    # Calculate feature for all audio files
    parser.add_argument('--dataset_file', type=str, required=True, help='csv file with all the dataset.')
    parser.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    parser.add_argument('--data_type', type=str, required=True, help='dev or eval.')
    # Parse arguments
    args = parser.parse_args()
    if args.data_type == 'dev':
        pack_audio_files_to_hdf5(args)
    elif args.data_type == 'eval':
        pack_audio_eval_files_to_hdf5(args)
    else:
        raise Exception('Incorrect arguments!')

