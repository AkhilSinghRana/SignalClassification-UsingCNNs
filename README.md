# Jammer Classification

This project aimes to provide a setup for Signal Classification/Detection of different Signal Spectogram signatures!

### Introduction

This project was started as an internal project from Airbus Defence and Space. The aim of the project was to verify the feasibility of classifying different Jammer signals using CNN. The solution in this repository is currently only tested on the real world Jammer signals processed using FFT transformation. The Data can't be provided because of the privacy policy. However, this code is free to be tested on any kind of Signals.

You need to preprocess, the signals to generate an image representation of it either using FFT or MFCC or any other signal processing algorithms. <b> Librosa </b>, --> https://librosa.github.io/librosa/ provides nice environment to do signal processing with python.


### Setup Instructions:
#### 1. Requirements

To reproduce the results from this repository, it is recommended to use virtual python environment and python version 3.7. Tensorflow version 2.0rc was used to build the models. Tensorflow_hub models were used for fine-tuning of the models.

Follow these simple steps to setup the dependencies:

```shell
git clone https://github.com/AkhilSinghRana/SignalClassification-UsingCNNs.git

cd SignalClassification-UsingCNNs

virtualenv env_name -p python3

source env_name/bin/activate #for linux


pip install -e .

 ```

Note*- The above code will setup all the required dependencies for you. Tested only on Linux


    
You are now ready to open the jupyter notebook for training and testing the pre trained models

#### 2. Testing/Loading model from checkpoint:

The checkpoint from my training is saved in Checkpoints [folder](./Checkpoints). Follow the instructions from the provided notebook. Test the trained agent and see the results.

``` jupyter notebook testReacherAgent.ipynb ```

#### 3. Train your own Agent:

Instructions for training your own agent is shown in below notebook.

``` jupyter notebook trainReacherAgent.ipynb  ```
 


### Results

The environment agent in this training was solved in 397 episodes! The algorithm used for training was DDPG. The results were achieved after a careful hyperparameter tuning, leading to a significant improvement. To read more about the algorithm, network architecture and hyper-Parameters settings read the [Report](./Report.pdf)

The training plot of the agent showing the scores improvement over the epsiodes is shown below.
![Scores](./Results/ScoresPlot.png)

