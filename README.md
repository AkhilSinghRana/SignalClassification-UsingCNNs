# Jammer Classification

This project aimes to do Classification, Detection and identifying different Signal Spectogram signatures!

### Introduction

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible. The solution in this repository is currently only tested on single agent, in future the solution to multiple agent environment will be added. 

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The task is episodic, and in order to solve the environment, the agent must get an average score of +30 over 100 consecutive episodes.

The trained agent from my training is shown below.
![TrainedAgent](./Results/TrainedAgent.gif)

You can learn more about the environment from the official Project instructions from Udacity [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control)

### Setup Instructions:
#### 1. Requirements

To reproduce the results from this repository, it is suggested to use virtual python environment and python version 3.6. Python 3.7 at the point of creating this repository does not support tensorflow=1.7 which is a dependency of unityagents package. Note* Python3.7 can still be used, if you know how to install pacakages from source, change requirements.txt and use latest version of tensorflow(tested with tf-v1.14). Follow these simple steps to setup the dependencies:

```shell
git clone https://github.com/AkhilSinghRana/ReinforcementLearning_Project-2_ContinuousControl.git

cd ReinforcementLearning_Project-2_ContinuousControl (cloned Repository root)

virtualenv env_name -p python3

source env_name/bin/activate #for linux


pip install -e .

 ```

Note*- The above code will setup all the required dependencies for you. Tested only on Linux


    
You are now ready to open the jupyter notebook for training and testing the Reacher agent!

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

