#!/usr/bin/env python
# -*- coding: utf-8 -*-
# DCASE 2022
# Task 1: low-complexity ASC with Multiple Devices
# Prepare data
# ---------------------------------------------
# Author: Irene Martin Morato
# Calculate features from 1 second files.


import dcase_util
from utils import *
from IPython import embed
import glob

__version_info__ = ('1', '0', '0')
__version__ = '.'.join(__version_info__)


def main(argv):
    # Read application default parameter file
    parameters = dcase_util.containers.DictContainer().load(
        filename='task1_features.yaml'
    )

    # Initialize application parameters
    param = dcase_util.containers.DCASEAppParameterContainer(
        parameters,
        path_structure={
            'FEATURE_EXTRACTOR': ['FEATURE_EXTRACTOR']
        }
    )

    # Handle application arguments
    args, overwrite = handle_application_arguments(
        app_parameters=param,
        raw_parameters=parameters,
        application_title='Task 1: prepare data',
        version=__version__
    )

    # Process parameters, this is done only after application argument handling in case
    # parameters where injected from command line.
    param.process()

    if args.parameter_set:
        # Check parameter set ids given as program arguments
        parameters_sets = args.parameter_set.split(',')

        # Check parameter_sets
        for set_id in parameters_sets:
            if not param.set_id_exists(set_id=set_id):
                raise ValueError('Parameter set id [{set_id}] not found.'.format(set_id=set_id))

    else:
        parameters_sets = [param.active_set()]

    # Get application mode
    if args.mode:
        application_mode = args.mode

    else:
        application_mode = 'dev'

    # Download only dataset if requested
    if args.dataset_path:

        # Make sure given path exists
        dcase_util.utils.Path().create(
            paths=args.dataset_path
        )

        for parameter_set in parameters_sets:
            # Set parameter set
            param['active_set'] = parameter_set
            param.update_parameter_set(parameter_set)

            if application_mode == 'eval':
                eval_parameter_set_id = param.active_set() + '_eval'
                if not param.set_id_exists(eval_parameter_set_id):
                    raise ValueError(
                        'Parameter set id [{set_id}] not found for eval mode.'.format(
                            set_id=eval_parameter_set_id
                        )
                    )

                # Change active parameter set
                param.update_parameter_set(eval_parameter_set_id)

            # Get dataset and initialize
            dcase_util.datasets.dataset_factory(
                dataset_class_name=param.get_path('dataset.parameters.dataset'),
                data_path=args.dataset_path,
            ).initialize().log()

        sys.exit(0)

    # Get overwrite flag
    if overwrite is None:
        overwrite = param.get_path('general.overwrite')

    # Make sure all system paths exists
    dcase_util.utils.Path().create(
        paths=list(param['path'].values())
    )

    # Get logging interface
    log = dcase_util.ui.ui.FancyLogger()

    # Log title
    log.title('DCASE2022 Task1 -- prepare data - feature extraction')
    log.line()

    if args.show_results:
        # Show evaluated systems
        show_results(param=param, log=log)
        sys.exit(0)

    if args.show_set_list:
        show_parameter_sets(param=param, log=log)
        sys.exit(0)

    # Create timer instance
    timer = dcase_util.utils.Timer()

    for parameter_set in parameters_sets:
        # Set parameter set
        param['active_set'] = parameter_set
        param.update_parameter_set(parameter_set)

        # Get dataset and initialize
        db_local = os.listdir(os.path.join(param.get_path('path.dataset'),'audio'))
        db_local.sort()


        if param.get_path('flow.feature_extraction'):
            # Feature extraction stage
            log.section_header('Feature Extraction')

            timer.start()

            processed_items = do_feature_extraction(
                db_local=db_local,
                param=param,
                log=log,
                overwrite=overwrite
            )

            timer.stop()

            log.foot(
                time=timer.elapsed(),
                item_count=len(processed_items)
            )

    return 0


def do_feature_extraction(db_local, param, log, overwrite=False):
    """Feature extraction stage

    Parameters
    ----------
    db_local : local files
        List of audio files

    param : dcase_util.containers.DCASEAppParameterContainer
        Application parameters

    log : dcase_util.ui.FancyLogger
        Logging interface

    overwrite : bool
        Overwrite data always
        Default value False

    Returns
    -------
    list of str

    """

    extraction_needed = False
    processed_files = []

    if overwrite:
        extraction_needed = True
    else:
        continue_item = 0
        for item_id, audio_filename in enumerate(db_local):
            # Get filename for feature data from audio filename
            feature_filename = dcase_util.utils.Path(
                path=audio_filename
            ).modify(
                path_base=param.get_path('path.application.feature_extractor'),
                filename_extension='.cpickle'
            )

            if not os.path.isfile(feature_filename):
                if continue_item == 0:
                    continue_item = item_id
                extraction_needed = True
                break
    # Prepare feature extractor
    if extraction_needed:

        method = param.get_path('feature_extractor.parameters.method', 'mel')
        if method == 'openl3':
            extractor = dcase_util.features.OpenL3Extractor(
                **param.get_path('feature_extractor.parameters', {})
            )

        elif method == 'edgel3':
            extractor = dcase_util.features.EdgeL3Extractor(
                **param.get_path('feature_extractor.parameters', {})
            )

        elif method == 'mel':
            extractor = dcase_util.features.MelExtractor(
                **param.get_path('feature_extractor.parameters', {})
            )

        else:
            raise ValueError('Unknown feature extractor method [{method}].'.format(method=method))
        # This is set to continue execution when stopped due to bad connection
        if continue_item != 0:
            for item_id in range(continue_item, len(db_local)):
                audio_filename = db_local[item_id]
                # Get filename for feature data from audio filename
                feature_filename = dcase_util.utils.Path(
                    path=audio_filename
                ).modify(
                    path_base=param.get_path('path.application.feature_extractor'),
                    filename_extension='.cpickle'
                )

                if not os.path.isfile(feature_filename) or overwrite:
                    log.line(
                        data='[{item: >5} / {total}] [{filename}]'.format(
                            item=item_id,
                            total=len(db_local),
                            filename=os.path.split(audio_filename)[1]
                        ),
                        indent=2
                    )

                    # Load audio data
                    audio = dcase_util.containers.AudioContainer().load(
                        filename=os.path.join(param.get_path('path.dataset'),'audio',audio_filename),
                        mono=True,
                        fs=param.get_path('feature_extractor.fs')
                    )

                    # Extract features and store them into FeatureContainer, and save it to the disk
                    dcase_util.containers.FeatureContainer(
                        data=extractor.extract(audio.data),
                        time_resolution=param.get_path('feature_extractor.hop_length_seconds')
                    ).save(
                        filename=feature_filename
                    )


        else:
            # Loop over all audio files in the current dataset and extract acoustic features for each of them.
            for item_id, audio_filename in enumerate(db_local):
                # Get filename for feature data from audio filename
                feature_filename = dcase_util.utils.Path(
                    path=audio_filename
                ).modify(
                    path_base=param.get_path('path.application.feature_extractor'),
                    filename_extension='.cpickle'
                )

                if not os.path.isfile(feature_filename) or overwrite:
                    log.line(
                        data='[{item: >5} / {total}] [{filename}]'.format(
                            item=item_id,
                            total=len(db_local),
                            filename=os.path.split(audio_filename)[1]
                        ),
                        indent=2
                    )

                    # Load audio data
                    audio = dcase_util.containers.AudioContainer().load(
                        filename=os.path.join(param.get_path('path.dataset'),'audio',audio_filename),
                        mono=True,
                        fs=param.get_path('feature_extractor.fs')
                    )

                    # Extract features and store them into FeatureContainer, and save it to the disk
                    dcase_util.containers.FeatureContainer(
                        data=extractor.extract(audio.data),
                        time_resolution=param.get_path('feature_extractor.hop_length_seconds')
                    ).save(
                        filename=feature_filename
                    )

        processed_files = glob.glob(param.get_path('path.application.feature_extractor')+"/*.cpickle")
    return processed_files


if __name__ == "__main__":
    sys.exit(main(sys.argv))