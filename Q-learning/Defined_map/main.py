from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns


import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

from functions import Params,Qlearning,EpsilonGreedy, run_env
from functions import postprocess , qtable_directions_map, plot_q_values_map, plot_states_actions_distribution
from functions import create_custom_frozenlake,generate_custom_map, custom_step #, CustomFrozenLake
from functions import tuple_to_grid_index


sns.set_theme()

params = Params(
    total_episodes=2000,
    learning_rate=0.8,
    gamma=0.95,
    epsilon=0.9,
    map_size=5,
    seed=123,
    is_slippery=False,
    n_runs=20,
    action_size=None,
    state_size=None,
    proba_frozen=0,
    savefig_folder=Path("./output/"),
)


# Set the seed
rng = np.random.default_rng(params.seed)

# Create the figure folder if it doesn't exists
params.savefig_folder.mkdir(parents=True, exist_ok=True)

# Example: Custom FrozenLake environment with size (12, 12) and holes at specific locations
custom_size = (5, 5)
custom_holes = [(1, 1), (1, 3), (4, 2)]
custom_goal = (2, 2)
custom_start = (0, 0)
custom_rewards = {67: 1}
env = create_custom_frozenlake(custom_size, custom_holes,custom_goal, custom_start)
# Monkey patch the step function with the custom_step function
env.original_step = env.step
env.step = custom_step
# env=CustomFrozenLake(custom_size, custom_holes, custom_goal, custom_start, custom_rewards, params.seed)

params = params._replace(action_size=env.action_space.n)
params = params._replace(state_size=env.observation_space.n)
print(f"Action size: {params.action_size}")
print(f"State size: {params.state_size}")

learner = Qlearning(
    learning_rate=params.learning_rate,
    gamma=params.gamma,
    state_size=params.state_size,
    action_size=params.action_size,
)

explorer = EpsilonGreedy(
    epsilon=params.epsilon,
    params=params
)

# map_sizes = [4, 7, 9, 11]
map_sizes = [12]
custom_size = (12, 12)
custom_holes = [(1, 2), (1, 7),
                (2, 2),(2, 3),(2, 4),(2, 7),(2, 8),(2, 9),
                (3, 2),(3, 3),(3, 4),(3, 7),(3, 8),(3, 9),
                (4, 2),(4, 3),(4, 4),(4, 7),(4, 8),
                (7, 3),(7, 4),(7, 7),(7, 8),(7, 9),(7, 10),
                (8, 2),(8, 3),(8, 4),(8, 7),(8, 8),(8, 9),
                (9, 2),(9, 3),(9, 4),(9, 7),(9, 8),(9, 9),
                (10, 4),(10, 8),(10, 9)]
custom_start = (5, 10)
custom_goal = (5, 6)
custom_rewards = {tuple_to_grid_index(custom_goal,map_sizes[0]): 3}
vertices = [(5, 6), (0, 11), (11, 11)]
obstacle_vertices= [[(1, 7), (1, 9), (4, 9),(4,7)],
                    [(7,7),(7,10),(10,10),(10,7)]]
be_sneaky=True
res_all = pd.DataFrame()
st_all = pd.DataFrame()

for map_size in map_sizes:
    env = create_custom_frozenlake(custom_size, custom_holes, custom_goal,custom_start)
    env.original_step = env.step
    env.step = custom_step
    # env = CustomFrozenLake(custom_size, custom_holes, custom_goal, custom_start, custom_rewards, params.seed)

    params = params._replace(action_size=env.action_space.n)
    params = params._replace(state_size=env.observation_space.n)
    env.action_space.seed(
        params.seed
    )  # Set the seed to get reproducible results when sampling the action space
    learner = Qlearning(
        learning_rate=params.learning_rate,
        gamma=params.gamma,
        state_size=params.state_size,
        action_size=params.action_size,
    )
    explorer = EpsilonGreedy(
        epsilon=params.epsilon,
        params=params
    )

    print(f"Map size: {map_size}x{map_size}")
    rewards, steps, episodes, qtables, all_states, all_actions = run_env(env,params, learner,explorer,
                                                                         custom_rewards,vertices,map_sizes[0],
                                                                         obstacle_vertices,be_sneaky,custom_goal)

    # Save the results in dataframes
    res, st = postprocess(episodes, params, rewards, steps, map_size)
    res_all = pd.concat([res_all, res])
    st_all = pd.concat([st_all, st])
    qtable = qtables.mean(axis=0)  # Average the Q-table between runs

    plot_states_actions_distribution(
        states=all_states, actions=all_actions, map_size=map_size, params=params
    )  # Sanity check
    plot_q_values_map(qtable, env, map_size,params)

    env.close()


