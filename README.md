The main goal of the CARLA Autonomous Driving Leaderboard is to evaluate the driving proficiency of autonomous agents in realistic traffic situations. The leaderboard serves as an open platform for the community to perform fair and reproducible evaluations, simplifying the comparison between different approaches.

Autonomous agents have to drive through a set of predefined routes. For each route, agents are initialized at a starting point and have to drive to a destination point. The agents will be provided with a description of the route. Routes will happen in a variety of areas, including freeways, urban scenes, and residential districts.

Agents will face multiple traffic situations based in the NHTSA typology, such as:

* Lane merging
* Lane changing
* Negotiations at traffic intersections
* Negotiations at roundabouts
* Handling traffic lights and traffic signs
* Coping with pedestrians, cyclists and other elements

The user can change the weather of the simulation, allowing the evaluation of the agent in a variety of weather conditions, including daylight scenes, sunset, rain, fog, and night, among others.

More information can be found [here](https://leaderboard.carla.org/)


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