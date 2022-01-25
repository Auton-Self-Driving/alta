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
# export TEAM_AGENT=$ALTA/agents/torch/multi_agent/leaderboard_agent_v2.py
# export TEAM_AGENT=$ALTA/agents/tf/team_code/auto_pilot.py
# export TEAM_AGENT=$ALTA/agents/tf/team_code/transfuser_autopilot.py
export TEAM_AGENT=$ALTA/agents/torch/multi_agent/offline_leaderboard_agent.py
export DEBUG_CHALLENGE=1
export RESUME=true
export CHECKPOINT_ENDPOINT=simulation_results_dvae_bt_seed2_noT.json
# export CHECKPOINT_ENDPOINT=simulation_results_bc_seed0.json
# export CHECKPOINT_ENDPOINT=simulation_results_dt_seed2.json
# export CHECKPOINT_ENDPOINT=simulation_results_tt_seed0.json
# export CHECKPOINT_ENDPOINT=simulation_results_iql_jan24.json
# export CHECKPOINT_ENDPOINT=sim_results_ckptDPPO1x8x1_LdbWG1kSG1kSteerScale0d5NoGoalLoadckptNoTerm_4868721_Nov060211AM52_devtest.json
# export CHECKPOINT_ENDPOINT=sim_results_ckptDPPO1x8x1_WG1kSG1kSteerScale0d5NoGoal5Obs15dimLdbNavi_7943856_Nov230612PM23_test.json
# export CHECKPOINT_ENDPOINT=sim_results_ckptDPPO1x8x1_WG1kSG1kSteerScale0d5NoGoal5Obs15dimLdbNavi_7540019_Nov231255PM00_test.json
# export CHECKPOINT_ENDPOINT=sim_results_ckptDPPO1x8x8_WG1kSG1kSteerScale0d5NoGoal5Obs15dim_12649216_Nov101019PM46_test.json
# export CHECKPOINT_ENDPOINT=sim_results_ckptDPPO1x8x8_WG1kSG1kSteerScale0d5NoGoal_8892417_Oct270136AM13_test.json
# export CHECKPOINT_ENDPOINT=sim_results_ckptDPPO1x8x1_WG1kSG1kSteerScale0d5NoGoal5Obs15dimLdbNavi_9067970_Nov240931AM50_test.json
# export CHECKPOINT_ENDPOINT=sim_results_ckptDPPO1x7x1_LdbWG1kSG1kSteerScale0d5NoGoal5Obs15dimPretrainWonly_2452288_Nov130823PM47_test.json

#python3 start_server.py &

python -u ${LEADERBOARD_ROOT}/leaderboard_evaluator.py \
--scenarios=${SCENARIOS}  \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--agent=${TEAM_AGENT} \
--debug=${DEBUG_CHALLENGE} \
--port=2000 \
--trafficManagerPort=4050 \
--checkpoint=${CHECKPOINT_ENDPOINT} \
# --resume=${RESUME} 
#--record=${RECORD_PATH} \
#--agent-config=${TEAM_CONFIG} \

