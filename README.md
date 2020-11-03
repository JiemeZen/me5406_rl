# ME5406 - Project 2

## Project Proposal:
https://docs.google.com/document/d/1RJDq4kErDUf3eswtbi8AToFeOjq-RR7MBGoKP2fwPBw/edit

## Instructions
1. Activate conda environment. ```conda activate your_env```
2. Install openai gym ```pip3 install gym```
3. Install Mujoco binaries and follow installation procedures. ```https://www.roboti.us/download/mujoco200_linux.zip```
4. Install mujoco-py ```pip3 install mujoco-py```
5. Install tensorflow 1.15 ```conda install tensorflow(-gpu)=1.15```

## Solo Xacro to URDF
1. Download the package from https://github.com/open-dynamic-robot-initiative/robot_properties_solo
2. Convert xacro to urdf using ```rosrun xacro xacro.py your.xacro > your.urdf\

## Viewing the XML
1. Go to: ```~/mujoco/mujoco200/bin``` first and run command below:
2. Inside the directory: ```mujoco200/bin$ ./simulate ~/path/me5406_rl/urdf/solo8.xml```
3. Under Control, each joints can be separately controlled
