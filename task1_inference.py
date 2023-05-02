#!/usr/bin/env python
# -*- coding: utf-8 -*-
# DCASE 2023
# Task 1: low-complexity ASC with Multiple Devices
# Inference baseline model
# ---------------------------------------------
# Author: Irene Martin Morato
# Based on the code dcase 2020 task A baseline from Toni Heittola
# Tampere University / Audio Research Group
# License: MIT

import dcase_util
from utils import *
import tensorflow as tf
from TAUUrbanAcousticScenes_2022_Mobile_DevelopmentSet import TAUUrbanAcousticScenes_2022_Mobile_EvaluationSet
from codecarbon import EmissionsTracker


__version_info__ = ('2', '0', '0')
__version__ = '.'.join(__version_info__)


def main():

    # Get logging interface
    log = dcase_util.ui.ui.FancyLogger()

    # Log title
    log.title('DCASE Task 1 -- low-complexity Acoustic Scene Classification')
    log.line()

    # Create timer instance
    timer = dcase_util.utils.Timer()


    # Get eval dataset and initialize
    db_eval = TAUUrbanAcousticScenes_2022_Mobile_EvaluationSet(
        storage_name='TAUUrbanAcousticScenes_2022_Mobile_EvaluationSet',
        data_path='dataset'
    )
    db_eval.prepare()

    # Get active folds
    active_folds = db_eval.folds( mode='full' )

    timer.start()
    
    
    # Path where the features are saved
    features_eval_path = '' 
    
    processed_items = do_testing_eval(
        db=db_eval,
        scene_labels=db_eval.scene_labels(),
        features_eval_path=features_eval_path,
        folds=active_folds,
        log=log
    )

    timer.stop()

    log.foot(
        time=timer.elapsed(),
        item_count=len(processed_items)
    )


    return 0


def do_testing_eval(db, scene_labels, features_eval_path, folds, log):

    ## CODE CARBON CODE STARTS HERE
    # Create code carbon instance
    path_codecarbon = 'codecarbon'
    tracker_test_eval = EmissionsTracker("DCASE Task 1 EVAL", output_dir=path_codecarbon)
    tracker_test_eval.start()

    processed_files = []

    # Loop over all cross-validation folds and test
    for fold in folds:
        log.line(
            data='Fold [{fold}]'.format(fold=fold),
            indent=2
        )

        # Get model filename
        fold_model_filename = 'model_task1.tflite'

        # Load the model into an interpreter
        interpreter = tf.lite.Interpreter(model_path=fold_model_filename)
        interpreter.allocate_tensors()

        # Get results filename
        fold_results_filename = os.path.join('res_fold_{fold}.csv'.format(fold=fold) )

        # Initialize results container
        res = dcase_util.containers.MetaDataContainer(
            filename=fold_results_filename
        )

        if not len(db.test(fold=fold)):
            raise ValueError('Dataset did not return any test files. Check dataset setup.')

        # Get input and output indexes for the interpreter
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]

        # Loop through all eval files from the evaluation fold
        test_files = db.test(fold=fold)

        features, _, filenames = get_data(features_eval_path, range(0, len(test_files)))
        
        for i, feat in enumerate(features):
            # Get feature filename
            filename = filenames[i]
            
            feat = np.expand_dims(feat, 0).astype(input_details["dtype"])
            input_data = tf.expand_dims(feat, -1)

            # Get network output
            interpreter.set_tensor(input_details['index'], input_data)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details['index'])
            probabilities = output.T
            # Clean up internal states.
            interpreter.reset_all_variables()

            
            probabilities = dcase_util.data.ProbabilityEncoder().collapse_probabilities(
                probabilities=probabilities,
                operator='sum',
                time_axis=1
            )

            # Binarization of the network output
            frame_decisions = dcase_util.data.ProbabilityEncoder().binarization(
                probabilities=probabilities,
                binarization_type='frame_max',
                threshold=0.5
            )

            estimated_scene_label = dcase_util.data.DecisionEncoder(
                label_list=scene_labels
            ).majority_vote(
                frame_decisions=frame_decisions
            )

            # Collect class wise probabilities and scale them between [0-1]
            class_probabilities = {}
            for scene_id, scene_label in enumerate(scene_labels):
                class_probabilities[scene_label] = probabilities[scene_id] / input_data.shape[0]

            res_data = {
                'filename': db.absolute_to_relative_path(filename),
                'scene_label': estimated_scene_label
            }
            # Add class class_probabilities
            res_data.update(class_probabilities)

            # Store result into results container
            res.append(
                res_data
            )
            
            processed_files.append(filename)

        ## Dump consumption
        tracker_test_eval.stop()
        log.section_header('Total energy kWh')
        log.line(
            data='Energy: {energy} kWh'.format(energy=tracker_test_eval._total_energy.kWh),
            indent=1
        )
        with open(os.path.join(path_codecarbon, "eval_energy_kwh.txt"), "w") as f:
            f.write(str(tracker_test_eval._total_energy.kWh))
        
        if not len(res):
            raise ValueError('No results to save.')

        # Save results container
        fields = ['filename', 'scene_label']
        fields += scene_labels

        res.save(fields=fields, csv_header=True)

    return processed_files



if __name__ == "__main__":
    sys.exit(main())
