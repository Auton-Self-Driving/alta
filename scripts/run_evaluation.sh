#!/bin/bash
export CARLA_ROOT=$HOME/carla910
export LEADERBOARD_ROOT=$ALTA/agents/tf/leaderboard

export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:$LEADERBOARD_ROOT
export PYTHONPATH=$PYTHONPATH:$ALTA/agents/tf/leaderboard/team_code
export PYTHONPATH=$PYTHONPATH:$ALTA/agents/tf/scenario_runner


export SCENARIOS=$LEADERBOARD_ROOT/data/all_towns_traffic_scenarios_public.json
export ROUTES=$LEADERBOARD_ROOT/data/routes_devtest.xml
export REPETITIONS=1
export CHALLENGE_TRACK_CODENAME=MAP
export TEAM_AGENT=$ALTA/agents/tf/alta_agent.py
export DEBUG_CHALLENGE=0

#python3 start_server.py &

python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
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
#--checkpoint=${CHECKPOINT_ENDPOINT} \
