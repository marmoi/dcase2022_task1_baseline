DCASE2022 - Task 1 A - Baseline systems
-------------------------------------

Author:
**Irene Martin**, *Tampere University* 
[Email](mailto:irene.martinmorato@tuni.fi). 
Adaptations from the original code DCASE2020 - Task 1 by
**Toni Heittola**, *Tampere University* 


Getting started
===============

1. Clone repository from [Github](https://github.com/marmoi/dcase2021_task1a_baseline).
2. Install requirements with command: `pip install -r requirements.txt`.
3. Extract features from the audio files previously downloaded `python prepare_data.py`.
4. Create a .h5 file with the extracted features. 
   - `python create_h5.py --dataset_file='/TAUUrbanAcousticScenes_2022_Mobile_DevelopmentSet/meta.csv' --workspace='path'`.   
5. Run the task specific application with default settings for model quantization `python task1.py` or  `./task1.py`


### Anaconda installation

To setup Anaconda environment for the system use following:

	conda create --name tf2-dcase python=3.6
	conda activate tf2-dcase
	conda install ipython
	conda install numpy
	conda install tensorflow-gpu=2.1.0
	conda install -c anaconda cudatoolkit
	conda install -c anaconda cudnn
	pip install librosa
	pip install absl-py==0.9.0
	pip install sed_eval
	pip install pyyaml==5.4
    pip install dcase_util
    pip install pandas
    pip install pyparsing==2.2.0


Introduction
============

This is the baseline system for the Low-Complexity Acoustic Scene Classification in Detection and Classification of Acoustic Scenes and Events 2022 (DCASE2022) challenge. The system is intended to provide a simple entry-level state-of-the-art approach that gives reasonable results. The baseline system is built on [dcase_util](https://github.com/DCASE-REPO/dcase_util) toolbox (>=version 0.2.16). 

Participants can build their own systems by extending the provided baseline system. 
The system is very simple, it does not handle dataset download or feature extraction, 
it loads the data from .h5 structure. The modular structure of the system enables 
participants to modify the system to their needs. 
The baseline system is a good starting point especially for the entry level researchers 
to familiarize themselves with the acoustic scene classification problem. 

If participants plan to publish their code to the DCASE community after the challenge,
building their approach on the baseline system could potentially make their code more
accessible to the community. DCASE organizers strongly encourage participants to share
their code in any form after the challenge.

### Data preparation
    |
    ├── task1_features.yaml   # Parameters for the prepare_data.py file
    ├── prepare_data.py       # Code to extract features from 1 second files
    └── create_h5.py          # Code to create the features_all.h5 file

Description
========

### Task 1 - Low-Complexity Acoustic Scene Classification 

[TAU Urban Acoustic Scenes 2022 Mobile Development dataset](https://zenodo.org/record/6337421) is used as development dataset for this task.

This subtask is concerned with the basic problem of acoustic scene classification, 
in which it is required to classify a test audio recording into one of ten known acoustic
scene classes. This task targets **generalization** properties of systems across a number
of different devices, and will use audio data recorded and simulated with a variety of devices.

Recordings in the dataset were made with three devices (A, B and C) that captured audio
simultaneously and 6 simulated devices (S1-S6). Each acoustic scene has 14400 segments
recorded with device A (main device) and 1080 segments of parallel audio each recorded
with devices B,C, and S1-S6. The dataset contains in total 64 hours of audio.
For a more detailed description see [DCASE Challenge task description](https://dcase.community/challenge2022/task-low-complexity-acoustic-scene-classification).

The task targets low complexity solutions for the classification problem in terms of
model size, and uses audio recorded with a single device (device A, 48 kHz / 24bit / stereo).
The data for the dataset was recorded in 10 acoustic scenes which were later grouped into three 
major classes used in this subtask. The dataset contains in total 40 hours of audio. 
For a more detailed description see [DCASE Challenge task description](https://dcase.community/challenge2022/task-low-complexity-acoustic-scene-classification).

The computational complexity will be measured in terms of parameter count and MMACs (million multiply-accumulate operations).
- Maximum number of parameters 128000, for **ALL** parameters, and the used variable type is fixed into **INT8**.
- Maximum number of MACS per inference: 30 MMAC (million MACs). The limit is approximated based on the computing power 
  of the target device class. The analysis segment length for the inference is 1 s.
  
See detailed description how to calculate model size from [DCASE Challenge task description](https://dcase.community/challenge2022/task-low-complexity-acoustic-scene-classification). 
Model calculation for TFLITE models is implemented using [nessi](https://github.com/AlbertoAncilotto/NeSsi) 

The task specific baseline system is implemented in file `task1.py`.

#### System description

The system implements a convolutional neural network (CNN) based approach, where log mel-band energies are first extracted 
for each 1-second signal, and a network consisting of two CNN layers and one fully connected layer is trained to assign 
scene labels to the audio signals. 
Model size of the baseline when using keras model quantization is 46.51 KB when using TFLite quantization 
and the MACS count is 29.23 M.


##### Parameters

###### Acoustic features

- Analysis frame 40 ms (50% hop size)
- Log mel-band energies (40 bands)

###### Neural network

- Input shape: 40 * 51 (1 second)
- Architecture:
  - CNN layer #1
    - 2D Convolutional layer (filters: 16, kernel size: 7) + Batch normalization + ReLu activation
  - CNN layer #2
    - 2D Convolutional layer (filters: 16, kernel size: 7) + Batch normalization + ReLu activation
    - 2D max pooling (pool size: (5, 5)) + Dropout (rate: 30%)
  - CNN layer #3
    - 2D Convolutional layer (filters: 32, kernel size: 7) + Batch normalization + ReLu activation
    - 2D max pooling (pool size: (4, 100)) + Dropout (rate: 30%)
  - Flatten
  - Dense layer #1
    - Dense layer (units: 100, activation: ReLu )
    - Dropout (rate: 30%)
  - Output layer (activation: softmax/sigmoid)
- Learning (epochs: 200, batch size: 16, data shuffling between epochs)
  - Optimizer: Adam (learning rate: 0.001)
- Model selection:
  - Approximately 30% of the original training data is assigned to validation set, split done so that training and validation sets do not have segments from same location and so that both sets have similar amount of data per city
  - Model performance after each epoch is evaluated on the validation set, and best performing model is selected
  
**Network summary**

    _________________________________________________________________
	Layer (type)                 Output Shape              Param #   
	=================================================================
	conv2d_1                     (None, 40, 51, 16)        800       
	_________________________________________________________________
	batch_normalization_1        (None, 40, 51, 16)        64        
	_________________________________________________________________
	activation_1                 (None, 40, 51, 16)        0         
	_________________________________________________________________
	conv2d_2                     (None, 40, 51, 16)        12560     
	_________________________________________________________________
	batch_normalization_2        (None, 40, 51, 16)        64        
	_________________________________________________________________
	activation_2                 (None, 40, 51, 16)        0         
	_________________________________________________________________
	max_pooling2d_1              (None, 8, 10, 16)         0         
	_________________________________________________________________
	dropout_1                    (None, 8, 10, 16)         0         
	_________________________________________________________________
	conv2d_3                     (None, 8, 10, 32)         25120     
	_________________________________________________________________
	batch_normalization_3        (None, 8, 10, 32)         128       
	_________________________________________________________________
	activation_3                 (None, 8, 10, 32)         0         
	_________________________________________________________________
	max_pooling2d_2              (None, 2, 1, 32)          0         
	_________________________________________________________________
	dropout_2                    (None, 2, 1, 32)          0         
	_________________________________________________________________
	flatten_1                    (None, 64)                0         
	_________________________________________________________________
	dense_1                      (None, 100)               6500      
	_________________________________________________________________
	dropout_3                    (None, 100)               0         
	_________________________________________________________________
	dense_2                      (None, 10)                1010      
	=================================================================

     Input shape                     : (None, 40, 51, 1)
     Output shape                    : (None, 10)

  
#### Results for development dataset

The cross-validation setup provided with the *TAU Urban Acoustic Scenes 2022 Mobile Development dataset* is used to evaluate the performance of the baseline system. Results are calculated using TensorFlow in GPU mode (using Nvidia Tesla V100 GPU card). Because results produced with GPU card are generally non-deterministic, the system was trained and tested 10 times, and mean and standard deviation of the performance from these 10 independent trials are shown in the results tables.
 

| Scene label       | Log Loss |   A   |   B   |   C   |   S1  |   S2  |   S3  |   S4  |   S5  |   S6  | Accuracy|  
| -------------     | -------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ------- | 
| Airport           | 1.534    | 1.165 | 1.439 | 1.475 | 1.796 | 1.653 | 1.355 | 1.608 | 1.734 | 1.577 | 39.4%   | 
| Bus               | 1.758    | 1.073 | 1.842 | 1.206 | 1.790 | 1.580 | 1.681 | 2.202 | 2.152 | 2.293 | 29.3%   |  
| Metro             | 1.382    | 0.898 | 1.298 | 1.183 | 2.008 | 1.459 | 1.288 | 1.356 | 1.777 | 1.166 | 47.9%   |  
| Metro station     | 1.672    | 1.582 | 1.641 | 1.833 | 2.010 | 1.857 | 1.613 | 1.643 | 1.627 | 1.247 | 36.0%   |  
| Park              | 1.448    | 0.572 | 0.513 | 0.725 | 1.615 | 1.130 | 1.678 | 2.314 | 1.875 | 2.613 | 58.9%   |  
| Public square     | 2.265    | 1.442 | 1.862 | 1.998 | 2.230 | 2.133 | 2.157 | 2.412 | 2.831 | 3.318 | 20.8%   |  
| Shopping mall     | 1.385    | 1.293 | 1.291 | 1.354 | 1.493 | 1.292 | 1.424 | 1.572 | 1.245 | 1.497 | 51.4%   |  
| Pedestrian street | 1.822    | 1.263 | 1.731 | 1.772 | 1.540 | 1.805 | 1.869 | 2.266 | 1.950 | 2.205 | 30.1%   |  
| Traffic street    | 1.025    | 0.830 | 1.336 | 1.023 | 0.708 | 1.098 | 1.147 | 0.957 | 0.634 | 1.489 | 70.6%   |  
| Tram              | 1.462    | 0.973 | 1.434 | 1.169 | 1.017 | 1.579 | 1.098 | 1.805 |2.176  | 1.903 | 44.6%   |  
| -------------     | -------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ------- |  
| Average          | **1.575**<br>(+/-0.018)| 1.109 | 1.439 | 1.374|  1.621 | 1.559 | 1.531 | 1.813|  1.800|  1.931 |  **42.9%**<br>(+/-0.770)|  
                                                                                

**Note:** The reported system performance is not exactly reproducible due to varying setups. However, you should be able obtain very similar results.


### Model size

 TFLite acoustic model

Tensor information (weights excluded, grouped by layer type):


| Id |               Tensor               |      Shape      | Size in RAM (B) |
|----|------------------------------------|-----------------|-----------------|
|  0 |           Identity_int8            |     (1, 10)     |              10 |
|  1 |         conv2d_input_int8          |  (1, 40, 51, 1) |           2,040 |
|  2 |     sequential/activation/Relu     | (1, 40, 51, 16) |          32,640 |
|  3 |    sequential/activation_1/Relu    | (1, 40, 51, 16) |          32,640 |
|  4 |    sequential/activation_2/Relu    |  (1, 8, 10, 32) |           2,560 |
| 13 |       sequential/dense/Relu        |     (1, 100)    |             100 |
| 14 |     sequential/dense_1/BiasAdd     |     (1, 10)     |              10 |
| 17 |  sequential/max_pooling2d/MaxPool  |  (1, 8, 10, 16) |           1,280 |
| 18 | sequential/max_pooling2d_1/MaxPool |  (1, 2, 1, 32)  |              64 |
| 19 |            conv2d_input            |  (1, 40, 51, 1) |           8,160 |
| 20 |              Identity              |     (1, 10)     |              40 |


Operator execution schedule:

|       Operator (output name)       | Tensors in memory (IDs) | Memory use (B) |       MACs |   Size |
|------------------------------------|-------------------------|----------------|------------|--------|
|         conv2d_input_int8          |         [1, 19]         |         10,200 |          0 |      0 |
|     sequential/activation/Relu     |          [1, 2]         |         34,680 |  1,599,360 |    848 |
|    sequential/activation_1/Relu    |          [2, 3]         |         65,280 | 25,589,760 | 12,608 |
|  sequential/max_pooling2d/MaxPool  |         [3, 17]         |         33,920 |     32,000 |      0 |
|    sequential/activation_2/Relu    |         [4, 17]         |          3,840 |  2,007,040 | 25,216 |
| sequential/max_pooling2d_1/MaxPool |         [4, 18]         |          2,624 |      2,560 |      0 |
|       sequential/dense/Relu        |         [13, 18]        |            164 |      3,200 |  6,800 |
|     sequential/dense_1/BiasAdd     |         [13, 14]        |            110 |      1,000 |  1,040 |
|           Identity_int8            |         [0, 14]         |             20 |          0 |      0 |
|              Identity              |         [0, 20]         |             50 |          0 |      0 |


Total MACs: 29,234,920
Total weight size: 46,512


Usage
=====

For the subtask there are two separate application (.py file):

- `task1.py`, DCASE2022 baseline for Task 1A, with TFLite model quantization


Code
====

The code is built on [dcase_util](https://github.com/DCASE-REPO/dcase_util) toolbox, see [manual for tutorials](https://dcase-repo.github.io/dcase_util/index.html). The machine learning part of the code is built on [TensorFlow (v2.1.0)](https://www.tensorflow.org/).

### File structure

      .
      ├── task1.py                                              # Baseline system for subtask A
      ├── task1.yaml                                            # Configuration file for task1a.py
      |
      ├── utils.py                                              # Common functions shared between tasks
      ├── TAUUrbanAcousticScenes_2022_Mobile_DevelopmentSet.py  # File for the dataset
      |
      ├── README.md                                             # This file
      └── requirements.txt                                      # External module dependencies

Changelog
=========

#### 2.0.0 / 2022-03-25


License
=======

This software is released under the terms of the [MIT License](https://github.com/toni-heittola/dcase2020_task1_baseline/blob/master/LICENSE).
