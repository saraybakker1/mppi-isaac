import gym
import numpy as np
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from mppiisaac.planner.mppi_isaac import MPPIisaacPlanner
import mppiisaac
from mppiisaac.planner.isaacgym_wrapper import IsaacGymWrapper, ActorWrapper
import hydra
from omegaconf import OmegaConf
import os
import torch
from urdfenvs.sensors.full_sensor import FullSensor
from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from mpscenes.goals.static_sub_goal import StaticSubGoal
from mppiisaac.priors.fabrics_panda import FabricsPandaPrior
from typing import Optional, List, Callable
from mppiisaac.utils.config_store import ExampleConfig
import yaml
from yaml.loader import SafeLoader

# MPPI to navigate a simple robot to a goal position

urdf_file = (
    os.path.dirname(os.path.abspath(__file__)) + "/../assets/urdf/panda_bullet/panda.urdf"
)


class JointSpaceGoalObjective(object):
    def __init__(self, cfg, device):
        self.nav_goal = torch.tensor(cfg.goal, device=cfg.mppi.device)

    def compute_cost(self, sim):
        pos = torch.cat(
            (
                sim.dof_state[:, 0].unsqueeze(1),
                sim.dof_state[:, 2].unsqueeze(1),
                sim.dof_state[:, 4].unsqueeze(1),
                sim.dof_state[:, 6].unsqueeze(1),
                sim.dof_state[:, 8].unsqueeze(1),
                sim.dof_state[:, 10].unsqueeze(1),
                sim.dof_state[:, 12].unsqueeze(1),
            ), 1)
        #dof_states = gym.acquire_dof_state_tensor(sim)
        return torch.clamp(
            torch.linalg.norm(pos - self.nav_goal, axis=1) - 0.05, min=0, max=1999
        )

class EndEffectorGoalObjective(object):
    def __init__(self, cfg, device):
        self.nav_goal = torch.tensor(cfg.goal, device=cfg.mppi.device)
        self.ort_goal = torch.tensor(cfg.mppi.goal_orientation, device=device)
        self.w_coll = 1.
        self.w_pos = 1.5
        self.w_ort = 0.5

    def compute_cost(self, sim):
        pos = sim.rigid_body_state[:, sim.robot_rigid_body_ee_idx, :3]
        ort = sim.rigid_body_state[:, sim.robot_rigid_body_ee_idx, 3:7]

        reach_cost = torch.linalg.norm(pos - self.nav_goal, axis=1)
        align_cost = torch.linalg.norm(ort - self.ort_goal, axis=1)

        # Collision avoidance with contact forces
        xyz_contatcs = torch.sum(torch.abs(torch.cat((sim.net_cf[:, 0].unsqueeze(1), sim.net_cf[:, 1].unsqueeze(1), sim.net_cf[:, 2].unsqueeze(1)), 1)),1)
        coll_cost = torch.sum(xyz_contatcs.reshape([sim.num_envs, int(xyz_contatcs.size(dim=0)/sim.num_envs)])[:, 1:sim.num_bodies], 1) # skip the first, it is the robot

        return reach_cost * self.w_pos + align_cost * self.w_ort + coll_cost * self.w_coll

class Action_to_Cost(object):
        def __init__(self, cfg, dynamics: Callable, running_cost: Callable): #objective:Callable
            self.horizon = cfg.mppi.horizon
            #retrieve these functions from mppi_isaac.py:
            self.running_cost=running_cost
            self.dynamics = dynamics
            # self.cfg = cfg
            self.tensor_args = {'device': cfg.mppi.device, 'dtype': torch.float32}
            self.dynamics = dynamics

        def _running_cost(self, state):
            # function from mppi.py
            return self.running_cost(state)

        def _dynamics(self, state, u, t=None):
            #function from mppi.py
            return self.dynamics(state, u, t=None)

        def compute_costs_over_horizon(self, actions_horizon, state, str_name):
            cost_horizon = torch.zeros([200, self.horizon], device=self.tensor_args['device'], dtype=self.tensor_args['dtype'])
            for t in range(1): #self.horizon):
                u = actions_horizon[t, :]
                cost = self._running_cost(state)
                print("Cost: ", str_name, cost)
                # cost_horizon[:, t] = cost
                # state, _ = self._dynamics(state, u, t)

def initialize_environment(cfg):
    """
    Initializes the simulation environment.

    Adds an obstacle and goal visualizaion to the environment and
    steps the simulation once.

    Params
    ----------
    render
        Boolean toggle to set rendering on (True) or off (False).
    """
    robots = [
        GenericUrdfReacher(urdf=urdf_file, mode="vel"),
    ]
    env: UrdfEnv = gym.make("urdf-env-v0", dt=0.01, robots=robots, render=cfg.render, observation_checking=False)

    # Set the initial position and velocity of the panda arm.
    env.reset()

    # add obstacle
    obst1Dict = {
        "type": "sphere",
        "geometry": {"position": [10.3, 0.3, 0.3], "radius": 0.1},
    }
    sphereObst1 = SphereObstacle(name="simpleSphere", content_dict=obst1Dict)
    env.add_obstacle(sphereObst1)

    obst2Dict = {
        "type": "sphere",
        "geometry": {"position": [10.3, 0.5, 0.6], "radius": 0.06},
    }
    sphereObst2 = SphereObstacle(name="simpleSphere", content_dict=obst2Dict)
    env.add_obstacle(sphereObst2)
    goal_dict = {
        "weight": 1.0,
        "is_primary_goal": True,
        "indices": [0, 1, 2],
        "parent_link": "panda_link0",
        "child_link": "panda_hand",
        "desired_position": cfg.goal,
        "epsilon": 0.05,
        "type": "staticSubGoal",
    }
    goal = StaticSubGoal(name="simpleGoal", content_dict=goal_dict)
    env.add_goal(goal)

    # sense both
    sensor = FullSensor(
        goal_mask=["position"],
        obstacle_mask=["position", "velocity", "size"],
        variance=0.0,
    )
    env.add_sensor(sensor, [0])
    env.set_spaces()
    return env

def set_planner(cfg):
    """
    Initializes the mppi planner for the panda arm.

    Params
    ----------
    goal_position: np.ndarray
        The goal to the motion planning problem.
    """
    objective = EndEffectorGoalObjective(cfg, cfg.mppi.device)
    #objective = JointSpaceGoalObjective(cfg, cfg.mppi.device)
    cfg.mppi.u_per_command = 0
    if cfg.mppi.use_priors == True:
        prior = FabricsPandaPrior(cfg)
    else:
        prior = None
    sim = create_simulator_isaac_gym(cfg)
    planner = MPPIisaacPlanner(cfg, sim, objective, prior, noise_sigma=cfg.mppi.noise_sigma)

    #create alternative planner:
    noise_sigma_alternative = [[1, 0., 0., 0., 0., 0., 0.],
                            [0., 1, 0., 0., 0., 0., 0.],
                            [0., 0., 1, 0., 0., 0., 0.],
                            [0., 0., 0., 1, 0., 0., 0.],
                            [0., 0., 0., 0., 1, 0., 0.],
                            [0., 0., 0., 0., 0., 1, 0.],
                            [0., 0., 0., 0., 0., 0., 1]]
        # [[1e-21, 0., 0., 0., 0., 0., 0.],
        #                     [0., 1e-21, 0., 0., 0., 0., 0.],
        #                     [0., 0., 1e-21, 0., 0., 0., 0.],
        #                     [0., 0., 0., 1e-21, 0., 0., 0.],
        #                     [0., 0., 0., 0., 1e-21, 0., 0.],
        #                     [0., 0., 0., 0., 0. ,1e-21, 0.],
        #                     [0., 0., 0., 0., 0., 0., 1e-21]]
    planner_alternative = MPPIisaacPlanner(cfg, sim, objective, prior, noise_sigma=noise_sigma_alternative)
    return planner, planner_alternative

def create_simulator_isaac_gym(cfg):
    actors = []
    for actor_name in cfg.actors:
        with open(f'{os.path.dirname(mppiisaac.__file__)}/../conf/actors/{actor_name}.yaml') as f:
            actors.append(ActorWrapper(**yaml.load(f, Loader=SafeLoader)))

    sim = IsaacGymWrapper(
        cfg.isaacgym,
        actors=actors,
        init_positions=cfg.initial_actor_positions,
        num_envs=cfg.mppi.num_samples,
    )
    return sim


@hydra.main(version_base=None, config_path="../conf", config_name="config_panda")
def run_panda_robot(cfg: ExampleConfig):
    """
    Set the gym environment, the planner and run panda robot example.
    The initial zero action step is needed to initialize the sensor in the
    urdf environment.

    Params
    ----------
    n_steps
        Total number of simulation steps.
    render
        Boolean toggle to set rendering on (True) or off (False).
    """
    # Note: Workaround to trigger the dataclasses __post_init__ method
    cfg = OmegaConf.to_object(cfg)

    #create list of possible goal orientations:
    cfg.mppi.goal_orientation = [1, 0, 0, 0]
    cfg.mppi.goal_orientations_alternative = [[0, 0, 0, 1]]

    env = initialize_environment(cfg)

    planner, planner_alternative = set_planner(cfg)
    action_to_cost_1 = Action_to_Cost(cfg=cfg, dynamics=planner.dynamics, running_cost=planner.running_cost)
    action_to_cost_2 = Action_to_Cost(cfg=cfg, dynamics=planner_alternative.dynamics, running_cost=planner_alternative.running_cost)

    action = np.zeros(7)
    ob, *_ = env.step(action)

    for _ in range(cfg.n_steps):
        # Calculate action with the fabric planner, slice the states to drop Z-axis [3] information.
        ob_robot = ob["robot_0"]
        obst = ob["robot_0"]["FullSensor"]['obstacles']
        q_current = ob_robot["joint_state"]["position"]
        qdot_current = ob_robot["joint_state"]["velocity"]
        actions = planner.compute_action(
            q=q_current,
            qdot=qdot_current,
            obst=obst
        )
        action = actions[0]
        action_to_cost_1.compute_costs_over_horizon(actions_horizon=actions, state=[q_current, qdot_current], str_name="planner")
        action_to_cost_2.compute_costs_over_horizon(actions_horizon=actions, state=[q_current, qdot_current], str_name="alternative planner")
        (
            ob,
            *_,
        ) = env.step(action)
    return {}


if __name__ == "__main__":
    res = run_panda_robot()
