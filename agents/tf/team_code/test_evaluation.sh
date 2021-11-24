#!/bin/bash
export CARLA_ROOT=$HOME/Documents/CARLA/carla910/
export LEADERBOARD_ROOT=$ALTA/leaderboard

export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:$LEADERBOARD_ROOT
export PYTHONPATH=$PYTHONPATH:$ALTA/agents/tf/team_code
export PYTHONPATH=$PYTHONPATH:$ALTA/scenario_runner
export PYTHONPATH=$PYTHONPATH:$ALTA
export PYTHONPATH=$PYTHONPATH:$ALTA/AdelaiDet

export SCENARIOS=$LEADERBOARD_ROOT/data/all_towns_traffic_scenarios_public.json
# export ROUTES=$LEADERBOARD_ROOT/data/routes_devtest.xml
export ROUTES=$LEADERBOARD_ROOT/data/routes_testing.xml
export REPETITIONS=1
# export CHALLENGE_TRACK_CODENAME=MAP
export CHALLENGE_TRACK_CODENAME=MAP
# export TEAM_AGENT=$ALTA/agents/tf/team_code/alta_agent.py
# export TEAM_AGENT=$ALTA/leaderboard/leaderboard/autoagents/npc_agent.py
# export TEAM_AGENT=$ALTA/agents/torch/multi_agent/leaderboard_agent.py
export TEAM_AGENT=$ALTA/agents/torch/multi_agent/leaderboard_agent_v2.py
export DEBUG_CHALLENGE=1
# export CHECKPOINT_ENDPOINT=sim_results_ckptDPPO1x8x1_LdbWG1kSG1kSteerScale0d5NoGoalLoadckptNoTerm_4868721_Nov060211AM52_devtest.json

#python3 start_server.py &

python -u ${LEADERBOARD_ROOT}/leaderboard_evaluator.py \
--scenarios=${SCENARIOS}  \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--agent=${TEAM_AGENT} \
--debug=${DEBUG_CHALLENGE} \
--port=2000 \
--trafficManagerPort=4050
#--record=${RECORD_PATH} \
#--resume=${RESUME}
#--agent-config=${TEAM_CONFIG} \
-checkpoint=${CHECKPOINT_ENDPOINT} \

