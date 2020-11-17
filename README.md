# ME5406 - Project 2

## Project Proposal
https://docs.google.com/document/d/1RJDq4kErDUf3eswtbi8AToFeOjq-RR7MBGoKP2fwPBw/edit

## Environment Installation Steps
1. Activate conda environment. ```$ conda activate your_env```
2. Install openai gym. ```$ pip install gym```
3. Install Mujoco 1.5 binaries and follow installation procedures. https://www.roboti.us/download/mjpro150_linux.zip
4. Install mujoco-py. ```$ pip install mujoco-py```
5. Install tensorflow 1.15 .```$ conda install tensorflow=1.15```

## Convert Solo8 urdf to xml
1. Download the package from https://github.com/open-dynamic-robot-initiative/robot_properties_solo
2. Convert xacro to urdf using ```rosrun xacro xacro.py your.xacro > your.urdf\

## Viewing the Robot XML
1. Go into mujoco 1.5 binary folder. ```$ cd ~/mujoco/mjpro150/bin```
2. Simulate robot model. ```$ ./simulate ~/path/me5406_rl/assets/solo8.xml```

## Training with our DDPG algorithm
Run ddpg_train.py to train the model. 1000 episodes will require about 1-2 hours. The trained model will be stored in ./trainedNet directory.</br>
```$ python ddpg_train.py --model_name myModel --training_episode 1000 --map ./assets/solo8.xml``` </br>
```$ python ddpg_train.py --model_name myModel```

## Evaluate the DDPG trained model
Run ddpg_evaluate.py to evaluate the model. Toggle verbose between 0 or 1 to view simulation informations.</br>
```$ python ddpg_evaluate.py --model_name ./trainedNet/myModel --verbose 0``` </br>
```$ python ddpg_evaluate.py --model_name ./trainedNet/myModel --verbose 1``` </br>

## Stable Baselines 
To compare our DDPG algorithm with other state-of-the-art algorithm, we used Stable Baselines library to train and evaluate the model. More information can be found here https://stable-baselines.readthedocs.io/en/master/
* Installation procedures
  * Update and install necessary dependencies. ```$ sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev```
  * Pip install baselines. ```$ pip install stable-baselines[mpi]```
* Training
  * Go to stable baselines directory. ```$ cd me5406_rl/stable_baselines```
  * Run train_baseline.py. Trained models will be saved under ./models folder. 
  * Train with saving intervals. One copy of the model will be saved at every intervals. ```$ python train_baseline.py --algo SAC --model_name myModel --training_step 200000 --step_interval 20000```
  * Train without intervals ```$ python train_baseline.py --algo SAC --model_name myModel --training_step 200000```
* Evaluation
  * Run evaluate_baseline.py to evaluate the model.
  * ```$ python evaluate_baseline.py --load ./models/myModel --verbose 0```
  * ```$ python evaluate_baseline.py --load ./models/myModel --verbose 1```

## Code References
1. DDPG written with tensorflow and keras. https://github.com/agakshat/tensorflow-ddpg
2. DDPG explained. https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b
3. DDPG paper. https://arxiv.org/pdf/1509.02971.pdf
4. Stable baselines. https://github.com/hill-a/stable-baselines
