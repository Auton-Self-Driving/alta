from carla.agent import Agent


class CarlaAgentWrapper(object):
    def __init__(self, agent):
        # Wrapper over an agent to interact with Carla benchmarks
        Agent.__init__(self)
        self.agent = agent
        
    def run_step(self, measurements, sensor_data, directions, target):
        """ API to interact with Carla benchmarks
        Inputs:
            - measurements: Carla object containing speed float
            - sensor_data: Carla object containing rgb image of shape (600, 800, 3)
            - directions: integer in {0,2,3,4,5} output by planner
            - target: not used (necessary for Carla API)
        Output:
            - control, Carla object containing steer, throttle and brake actions
        """
        # Extract input tensors from raw observation
        obs = self.agent.process_observation(measurements, sensor_data, directions)
        
        # Compute action from input tensors
        action = self.agent.get_action(obs, eval_mode=True)
        
        # Get control for Carla simulator from action tensor
        control = self.agent.get_control(action)
        
        return control