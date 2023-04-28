import os
import collections
from dcase_util.datasets import AcousticSceneDataset
from dcase_util.containers import MetaDataContainer, DictContainer

# =====================================================
# DCASE 2022
# =====================================================


class TAUUrbanAcousticScenes_2022_Mobile_DevelopmentSet(AcousticSceneDataset):
    """TAU Urban Acoustic Scenes 2022 Mobile Development dataset

    This dataset is used in DCASE2022 - Task 1, Acoustic scene classification/ Development
    For the development part, where the features have been calculated already and save them in a h5 file
    """

    def __init__(self,
                 storage_name='TAU-urban-acoustic-scenes-2022-mobile-development',
                 data_path=None,
                 included_content_types=None,
                 **kwargs):
        """
        Constructor

        Parameters
        ----------

        storage_name : str
            Name to be used when storing dataset on disk
            Default value 'TAU-urban-acoustic-scenes-2022-mobile-development'

        data_path : str
            Root path where the dataset is stored. If None, os.path.join(tempfile.gettempdir(), 'dcase_util_datasets')
            is used.
            Default value None

        included_content_types : list of str or str
            Indicates what content type should be processed. One or multiple from ['all', 'audio', 'meta', 'code',
            'documentation']. If None given, ['all'] is used. Parameter can be also comma separated string.
            Default value None

        """

        kwargs['included_content_types'] = included_content_types
        kwargs['data_path'] = data_path
        kwargs['storage_name'] = storage_name
        kwargs['dataset_group'] = 'scene'
        kwargs['dataset_meta'] = {
            'authors': 'Toni Heittola, Annamaria Mesaros, and Tuomas Virtanen',
            'title': 'TAU Urban Acoustic Scenes 2022 Mobile, development dataset',
            'url': None,
            'audio_source': 'Field recording',
            'audio_type': 'Natural/Synthetic',
            'audio_recording_device_model': 'Various',
            'microphone_model': 'Various',
            'licence': 'free non-commercial'
        }
        kwargs['crossvalidation_folds'] = 1
        kwargs['evaluation_setup_file_extension'] = 'csv'
        kwargs['meta_filename'] = 'meta.csv'

        kwargs['audio_paths'] = [
            'audio'
        ]
        super(TAUUrbanAcousticScenes_2022_Mobile_DevelopmentSet, self).__init__(**kwargs)

    def process_meta_item(self, item, absolute_path=True, **kwargs):
        """Process single meta data item

        Parameters
        ----------
        item :  MetaDataItem
            Meta data item

        absolute_path : bool
            Convert file paths to be absolute
            Default value True

        """

        if absolute_path:
            item.filename = self.relative_to_absolute_path(item.filename)

        else:
            item.filename = self.absolute_to_relative_path(item.filename)

        if not item.identifier:
            item.identifier = '-'.join(os.path.splitext(os.path.split(item.filename)[-1])[0].split('-')[1:-2])

        if not item.source_label:
            item.source_label = os.path.splitext(os.path.split(item.filename)[-1])[0].split('-')[-1]

    def prepare(self):
        """Prepare dataset for the usage.

        Returns
        -------
        self

        """

        if not self.meta_container.exists():
            meta_data = collections.OrderedDict()
            for fold in self.folds():
                # Read train files in
                fold_data = MetaDataContainer(
                    filename=self.evaluation_setup_filename(
                        setup_part='train',
                        fold=fold
                    )
                ).load()

                # Read eval files in
                fold_data += MetaDataContainer(
                    filename=self.evaluation_setup_filename(
                        setup_part='evaluate',
                        fold=fold
                    )
                ).load()

                # Process, make sure each file is included only once.
                for item in fold_data:
                    if item.filename not in meta_data:
                        self.process_meta_item(
                            item=item,
                            absolute_path=False
                        )

                        meta_data[item.filename] = item

            # Save meta
            MetaDataContainer(list(meta_data.values())).save(
                filename=self.meta_file
            )

            # Load meta and cross validation
            self.load()

        return self
    
class TAUUrbanAcousticScenes_2022_Mobile_EvaluationSet(AcousticSceneDataset):
    """TAU Urban Acoustic Scenes 2022 Mobile Evaluation dataset

    This dataset is used in DCASE2022 - Task 1, Acoustic scene classification / Evaluation
    For the evaluation part, where the features have been calculated already and save them in a h5 file
    """

    def __init__(self,
                 storage_name='TAU-urban-acoustic-scenes-2022-mobile-evaluation',
                 data_path=None,
                 included_content_types=None,
                 **kwargs):
        """
        Constructor

        Parameters
        ----------

        storage_name : str
            Name to be used when storing dataset on disk
            Default value 'TAU-urban-acoustic-scenes-2022-mobile-evaluation'

        data_path : str
            Root path where the dataset is stored. If None, os.path.join(tempfile.gettempdir(), 'dcase_util_datasets')
            is used.
            Default value None

        included_content_types : list of str or str
            Indicates what content type should be processed. One or multiple from ['all', 'audio', 'meta', 'code',
            'documentation']. If None given, ['all'] is used. Parameter can be also comma separated string.
            Default value None

        """

        kwargs['included_content_types'] = included_content_types
        kwargs['data_path'] = data_path
        kwargs['storage_name'] = storage_name
        kwargs['dataset_group'] = 'scene'
        kwargs['dataset_meta'] = {
            'authors': 'Toni Heittola, Annamaria Mesaros, and Tuomas Virtanen',
            'title': 'TAU Urban Acoustic Scenes 2022 Mobile, evaluation dataset',
            'url': None,
            'audio_source': 'Field recording',
            'audio_type': 'Natural/Synthetic',
            'audio_recording_device_model': 'Various',
            'microphone_model': 'Various',
            'licence': 'free non-commercial'
        }
        kwargs['crossvalidation_folds'] = 1
        kwargs['evaluation_setup_file_extension'] = 'csv'
        kwargs['meta_filename'] = 'meta.csv'

        kwargs['audio_paths'] = [
            'audio'
        ]
        super(TAUUrbanAcousticScenes_2022_Mobile_EvaluationSet, self).__init__(**kwargs)

    def process_meta_item(self, item, absolute_path=True, **kwargs):
        """Process single meta data item

        Parameters
        ----------
        item :  MetaDataItem
            Meta data item

        absolute_path : bool
            Convert file paths to be absolute
            Default value True

        """

        if absolute_path:
            item.filename = self.relative_to_absolute_path(item.filename)

        else:
            item.filename = self.absolute_to_relative_path(item.filename)


    def prepare(self):
        """Prepare dataset for the usage.

        Returns
        -------
        self

        """

        if not self.meta_container.exists():
            meta_data = collections.OrderedDict()
            for fold in self.folds():
                # Read train files in
                fold_data = MetaDataContainer(
                    filename=self.evaluation_setup_filename(
                        setup_part='train',
                        fold=fold
                    )
                ).load()

                # Read eval files in
                fold_data += MetaDataContainer(
                    filename=self.evaluation_setup_filename(
                        setup_part='evaluate',
                        fold=fold
                    )
                ).load()

                # Process, make sure each file is included only once.
                for item in fold_data:
                    if item.filename not in meta_data:
                        self.process_meta_item(
                            item=item,
                            absolute_path=False
                        )

                        meta_data[item.filename] = item

            # Save meta
            MetaDataContainer(list(meta_data.values())).save(
                filename=self.meta_file
            )

            # Load meta and cross validation
            self.load()

        return self

    def load_crossvalidation_data(self):
        """Load cross-validation into the container.
        Returns
        -------
        self
        """

        # Reset cross validation data and insert 'all_data'
        if self.meta_container:
            # Meta data is available
            self.crossvalidation_data = DictContainer({
                'train': {
                    'all_data': self.meta_container
                },
                'test': {
                    'all_data': self.meta_container
                },
                'evaluate': {
                    'all_data': self.meta_container
                },
            })

        else:
            # No meta data available, load data from evaluation setup files (if they exists).
            self.crossvalidation_data = DictContainer({
                'train': {
                    'all_data': MetaDataContainer()
                },
                'test': {
                    'all_data': MetaDataContainer()
                },
                'evaluate': {
                    'all_data': MetaDataContainer()
                },
            })

            test_filename = self.evaluation_setup_filename(setup_part='test', fold=1)
            evaluate_filename = self.evaluation_setup_filename(setup_part='evaluate', fold=1)

            if os.path.isfile(test_filename):
                # Testing data exists, load and process it
                self.crossvalidation_data['test']['all_data'] = self.process_meta_container(
                    container=MetaDataContainer(filename=test_filename).load()
                )

                # Process items
                for item in self.crossvalidation_data['test']['all_data']:
                    self.process_meta_item(item=item)

            if os.path.isfile(evaluate_filename):
                # Evaluation data exists, load and process it
                self.crossvalidation_data['evaluate']['all_data'] = self.process_meta_container(
                    container=MetaDataContainer(filename=evaluate_filename).load()
                )

                # Process items
                for item in self.crossvalidation_data['evaluate']['all_data']:
                    self.process_meta_item(item=item)

        for crossvalidation_set in list(self.crossvalidation_data.keys()):
            for item in self.crossvalidation_data[crossvalidation_set]['all_data']:
                self.process_meta_item(item=item)

        return self

    def scene_labels(self):
        """List of unique scene labels in the meta data.
        Returns
        -------
        list
            List of scene labels in alphabetical order.
        """

        return ['airport',
                'bus',
                'metro',
                'metro_station',
                'park',
                'public_square',
                'shopping_mall',
                'street_pedestrian',
                'street_traffic',
                'tram']