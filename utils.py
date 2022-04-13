#!/usr/bin/env python
# -*- coding: utf-8 -*-
import dcase_util
import sys
import os
import argparse
import textwrap
import numpy as np
import pandas as pd
import h5py
import tensorflow as tf

def handle_application_arguments(app_parameters, raw_parameters, application_title='', version=''):
    """Handle application arguments

    Parameters
    ----------
    app_parameters : dcase_util.containers.DCASEAppParameterContainer
        Application parameters

    raw_parameters : dict
        Application parameters in dict format

    application_title : str
        Application title
        Default value ''

    version : str
        Application version
        Default value ''

    Returns
    -------
    nothing


    """

    parser = argparse.ArgumentParser(
        prefix_chars='-+',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            '''\
            DCASE 2021 
            {app_title}
            Baseline system
            ---------------------------------------------            
            Author:  Toni Heittola ( toni.heittola@tuni.fi )
            Tampere University / Audio Research Group
            '''.format(app_title=application_title)
        )
    )

    # Setup argument handling
    parser.add_argument(
        '-m', '--mode',
        choices=('dev', 'eval'),
        default=None,
        help="Selector for application operation mode",
        required=False,
        dest='mode',
        type=str
    )

    # Application parameter modification
    parser.add_argument(
        '-s', '--parameter_set',
        help='Parameter set id, can be comma separated list',
        dest='parameter_set',
        required=False,
        type=str
    )

    parser.add_argument(
        '-p', '--param_file',
        help='Parameter file override',
        dest='parameter_override',
        required=False,
        metavar='FILE',
        type=dcase_util.utils.argument_file_exists
    )

    # Specific actions
    parser.add_argument(
        '--overwrite',
        help='Overwrite mode',
        dest='overwrite',
        action='store_true',
        required=False
    )

    parser.add_argument(
        '--download_dataset',
        help='Download dataset to given path and exit',
        dest='dataset_path',
        required=False,
        type=str
    )

    # Output
    parser.add_argument(
        '-o', '--output',
        help='Output file',
        dest='output_file',
        required=False,
        type=str
    )

    # Show information
    parser.add_argument(
        '--show_parameters',
        help='Show active application parameter set',
        dest='show_parameters',
        action='store_true',
        required=False
    )

    parser.add_argument(
        '--show_sets',
        help='List of available parameter sets',
        dest='show_set_list',
        action='store_true',
        required=False
    )

    parser.add_argument(
        '--show_results',
        help='Show results of the evaluated system setups',
        dest='show_results',
        action='store_true',
        required=False
    )

    # Application information
    parser.add_argument(
        '-v', '--version',
        help='Show version number and exit',
        action='version',
        version='%(prog)s ' + version
    )

    # Select random seed
    parser.add_argument(
        '-r', '--r_seed',
        help='Specify random seed',
        dest='seed',
        required=False,
        type=int
    )

    # Parse arguments
    args = parser.parse_args()

    if args.parameter_override:
        # Override parameters from a file
        app_parameters.override(override=args.parameter_override)

    overwrite = None
    if args.overwrite:
        overwrite = True

    if args.show_parameters:
        # Process parameters, and clean up parameters a bit for showing

        if args.parameter_set:
            # Check parameter set ids given as program arguments
            parameters_sets = args.parameter_set.split(',')

            for parameter_set in parameters_sets:
                # Set parameter set
                param_current = dcase_util.containers.DCASEAppParameterContainer(
                    raw_parameters,
                    path_structure={
                        'FEATURE_EXTRACTOR': ['FEATURE_EXTRACTOR'],
                        'FEATURE_NORMALIZER': ['FEATURE_EXTRACTOR'],
                        'LEARNER': ['DATA_PROCESSING_CHAIN', 'LEARNER'],
                        'RECOGNIZER': ['DATA_PROCESSING_CHAIN', 'LEARNER', 'RECOGNIZER'],
                    }
                )
                if args.parameter_override:
                    # Override parameters from a file
                    param_current.override(override=args.parameter_override)

                param_current.process(
                    create_paths=False,
                    create_parameter_hints=False
                )

                param_current['active_set'] = parameter_set
                param_current.update_parameter_set(parameter_set)
                del param_current['sets']
                del param_current['defaults']
                for section in param_current:
                    if section.endswith('_method_parameters'):
                        param_current[section] = {}

                param_current.log()
        else:
            app_parameters.process(
                create_paths=False,
                create_parameter_hints=False
            )
            del app_parameters['sets']
            del app_parameters['defaults']
            for section in app_parameters:
                if section.endswith('_method_parameters'):
                    app_parameters[section] = {}

            app_parameters.log()
        sys.exit(0)

    return args, overwrite


def save_system_output(db, folds, param, log, output_file, mode='dcase'):
    """Save system output

    Parameters
    ----------
    db : dcase_util.dataset.Dataset
        Dataset

    folds : list
        List of active folds

    param : dcase_util.containers.DCASEAppParameterContainer
        Application parameters

    log : dcase_util.ui.FancyLogger
        Logging interface

    output_file : str

    mode : str
        Output mode, possible values ['dcase', 'leaderboard']
        Default value 'dcase'

    Returns
    -------
    nothing

    """

    # Initialize results container
    all_res = dcase_util.containers.MetaDataContainer(
        filename=output_file
    )

    # Loop over all cross-validation folds and collect results
    for fold in folds:
        # Get results filename
        fold_results_filename = os.path.join(
            param.get_path('path.application.recognizer'),
            'res_fold_{fold}.csv'.format(fold=fold)
        )

        if os.path.isfile(fold_results_filename):
            # Load results container
            res = dcase_util.containers.MetaDataContainer().load(
                filename=fold_results_filename
            )
            all_res += res

        else:
            raise ValueError(
                'Results output file does not exists [{fold_results_filename}]'.format(
                    fold_results_filename=fold_results_filename
                )
            )

    if len(all_res) == 0:
        raise ValueError(
            'There are no results to output into [{output_file}]'.format(
                output_file=output_file
            )
        )

    # Convert paths to relative to the dataset root
    for item in all_res:
        item.filename = db.absolute_to_relative_path(item.filename)

        if mode == 'leaderboard':
            item['Id'] = os.path.splitext(os.path.split(item.filename)[-1])[0]
            item['Scene_label'] = item.scene_label

    if mode == 'leaderboard':
        all_res.save(fields=['Id', 'Scene_label'], delimiter=',')

    else:
        fields = ['filename', 'scene_label']
        fields += db.scene_labels()
        all_res.save(fields=fields, csv_header=True)

    log.line('System output saved to [{output_file}]'.format(output_file=output_file), indent=2)
    log.line()


def show_general_information(parameter_set, active_folds, param, db, log):
    """Show application general information

    Parameters
    ----------
    parameter_set : str
        Dataset

    active_folds : list
        List of active folds

    param : dcase_util.containers.DCASEAppParameterContainer
        Application parameters

    db : dcase_util.dataset.Dataset
        Dataset

    log : dcase_util.ui.FancyLogger
        Logging interface

    Returns
    -------
    nothing

    """

    log.section_header('General information')
    log.line('Parameter set', indent=2)
    log.data(field='Set ID', value=parameter_set, indent=4)
    log.data(field='Set description', value=param.get('description'), indent=4)

    log.line('Application', indent=2)
    log.data(field='Overwrite', value=param.get_path('general.overwrite'), indent=4)

    log.data(field='Dataset', value=db.storage_name, indent=4)
    log.data(field='Active folds', value=active_folds, indent=4)
    log.line()
    log.foot()


def show_results(param, log):
    """Show system evaluation results

    Parameters
    ----------
    param : dcase_util.containers.DCASEAppParameterContainer
        Application parameters

    log : dcase_util.ui.FancyLogger
        Logging interface

    Returns
    -------
    nothing

    """

    eval_path = param.get_path('path.application.evaluator')

    eval_files = dcase_util.utils.Path().file_list(path=eval_path, extensions='yaml')

    eval_data = {}
    for filename in eval_files:
        data = dcase_util.containers.DictContainer().load(filename=filename)
        set_id = data.get_path('parameters.set_id')
        if set_id not in eval_data:
            eval_data[set_id] = {}

        params_hash = data.get_path('parameters._hash')

        if params_hash not in eval_data[set_id]:
            eval_data[set_id][params_hash] = data

    log.section_header('Evaluated systems')
    log.line()
    log.row_reset()
    log.row(
        'Set ID', 'Mode', 'Accuracy', 'Description', 'Parameter hash',
        widths=[25, 10, 11, 45, 35],
        separators=[False, True, True, True, True],
        types=['str25', 'str10', 'float1_percentage', 'str', 'str']
    )
    log.row_sep()
    for set_id in sorted(list(eval_data.keys())):
        for params_hash in eval_data[set_id]:
            data = eval_data[set_id][params_hash]
            desc = data.get_path('parameters.description')
            application_mode = data.get_path('application_mode', '')
            log.row(
                set_id,
                application_mode,
                data.get_path('overall_accuracy') * 100.0,
                desc,
                params_hash
            )
    log.line()
    sys.exit(0)


def show_parameter_sets(param, log):
    """Show available parameter sets

    Parameters
    ----------
    param : dcase_util.containers.DCASEAppParameterContainer
        Application parameters

    log : dcase_util.ui.FancyLogger
        Logging interface

    Returns
    -------
    nothing

    """

    log.section_header('Parameter sets')
    log.line()
    log.row_reset()
    log.row(
        'Set ID', 'Description',
        widths=[50, 70],
        separators=[True, False],
    )
    log.row_sep()
    for set_id in param.set_ids():
        current_parameter_set = param.get_set(set_id=set_id)

        if current_parameter_set:
            desc = current_parameter_set.get('description', '')
        else:
            desc = ''

        log.row(
            set_id,
            desc
        )

    log.line()


def float32_to_int16(x):
    if np.max(np.abs(x)) > 1.:
        x /= np.max(np.abs(x))
    return (x * 32767.).astype(np.int16)


def int16_to_float32(x):
    return (x / 32767.).astype(np.float32)


def create_one_hot_encoding(word, unique_words):
    """Creates an one-hot encoding of the `word` word, based on the\
    list of unique words `unique_words`.
    """
    to_return = np.zeros((len(unique_words)))
    to_return[unique_words.index(word)] = 1
    return to_return

# Function to get indexes for the different splits (Training, testing)
# Input gets the metadata file from the TAU DATASET and the filenames corresponding to the split
# Output list of indexes
def get_index(param, file_list):
    meta_file = pd.read_csv(os.path.join(param.get_path('path.dataset'), param.get_path('dataset.parameters.dataset'),'meta.csv'), sep='\t')
    name_files = meta_file['filename'].to_list()
    index_file = []
    for file in file_list:
        index_file.append(name_files.index('audio'+file['filename'].split('audio')[1]))
    return index_file


# Function to read HDF5 and getting the correct split
def get_data(hdf5_path, index_file):
    with h5py.File(hdf5_path, 'r') as hf:
        features = int16_to_float32(hf['features'][index_file])
        labels = [f.decode() for f in hf['scene_label'][index_file]]
        audio_name = [f.decode() for f in hf['filename'][index_file]]
    return features, labels, audio_name


def get_specialindex(param, file_list):
    meta_file = pd.read_csv(os.path.join(param.get_path('path.dataset'), param.get_path('dataset.parameters.dataset'),'meta.csv'), sep='\t')
    name_files = meta_file['filename'].to_list()
    index_file = []
    for file in file_list:
        index_file.append(name_files.index('audio/'+file['data']['filename']))
    return index_file


def getIndex(param, file_list):
    meta_file = pd.read_csv(os.path.join(param.get_path('path.dataset'), param.get_path('dataset.parameters.dataset'),'meta.csv'), sep='\t')
    name_files = meta_file['filename'].to_list()
    index_file = [name_files.index('audio/'+file) for file in file_list]
    return index_file


def load_data(unique_words, files, param, split):
    Y = []
    index_files = getIndex(param, files)
    # Get files from h5
    features, labels, _ = get_data(param.get_path('features.path'), index_files)
    for lab in labels:
        if split == 'Train':
            Y.append(smooth_labels(create_one_hot_encoding(lab, unique_words)))
        else:
            Y.append(create_one_hot_encoding(lab, unique_words))

    data_size = {
        'data': features[0].shape[0],
        'time': features[0].shape[1],
    }
    return features, Y, data_size


def smooth_labels(labels, factor=0.1):
    labels *= (1 - factor)
    labels += (factor / len(labels))
    return labels


class BatchGenerator():
    def __init__(self,
                 hdf5_path,
                 batch_size=32):
        self.hdf5_path = hdf5_path
        self.batch_size = batch_size

    def __call__(self):
        index_in_hdf5 = np.arange(self.batch_size)
        with h5py.File(self.hdf5_path, 'r') as hf:
            features = int16_to_float32(hf['features'][index_in_hdf5])
        for feat in features:
            yield ([tf.expand_dims(tf.expand_dims(feat,-1), 0)])