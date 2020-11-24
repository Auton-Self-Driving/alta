# agents

* tf - Contains latest code in Tensorflow 1.10.0

* torch - Contains old code in PyTorch Code for DDPG/SAC/TD3. This code is retained to enable running other algorithms but needs to be integrated with the latest code in agents/tf and environment.


####################################################################################################################################
Instructions:

1) Install the following modules:
pip install py-trees==0.8.3
pip install ephem

2) Start carla server in a separate tmux session using:
$CARLA_9_4_PATH/CarlaUE4.sh -world-port=2000 -gpu=1
(Note: If you kill your previous run forcefully, they you have to restart the server again to avoid timeout errors)

3) Go to leaderboard/scripts and make any required changes such as path, required routes and scenarios

4) Run :
bash run_evaluation.sh

5) All our changes have to be made in alta_agent.py file located in agents/tf. Currently, I wrote a template to check if the all the sizes are as expected and the code runs without any erros. You can start filling in the required specified functions.

6) You can see the results in simiulation_results.json file in leaderboard/scripts folder