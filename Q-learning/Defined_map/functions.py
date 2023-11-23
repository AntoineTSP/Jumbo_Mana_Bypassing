from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import gymnasium as gym


class Params(NamedTuple):
    total_episodes: int  # Total episodes
    learning_rate: float  # Learning rate
    gamma: float  # Discounting rate
    epsilon: float  # Exploration probability
    map_size: int  # Number of tiles of one side of the squared environment
    seed: int  # Define a seed so that we get reproducible results
    is_slippery: bool  # If true the player will move in intended direction with probability of 1/3 else will move in either perpendicular direction with equal probability of 1/3 in both directions
    n_runs: int  # Number of runs
    action_size: int  # Number of possible actions
    state_size: int  # Number of possible states
    proba_frozen: float  # Probability that a tile is frozen
    savefig_folder: Path  # Root folder where plots are saved

def tuple_to_grid_index(tuple, map_size):
    return tuple[0]*map_size +tuple[1]

def grid_index_to_tuple(index,map_size):
    return index//map_size,index%map_size
def distance_two_point(tuple1,tuple2):
    return np.linalg.norm(np.array(tuple1) - np.array(tuple2))/10

def create_custom_frozenlake(size, hole_locations, goal_location, start_location):
    custom_map = generate_custom_map(size, hole_locations,goal_location, start_location)
    env = gym.make('FrozenLake-v1', desc=custom_map, is_slippery=False, render_mode="rgb_array")
    return env


def generate_custom_map(size, hole_locations, goal_location, start_location):
    custom_map = np.full((size[0], size[1]), 'F', dtype='str')  # Initialize with 'H' for holes

    for hole_loc in hole_locations:
        custom_map[hole_loc[0], hole_loc[1]] = 'H'

    custom_map[goal_location[0], goal_location[1]] = 'G'  # Goal
    custom_map[start_location[0], start_location[1]] = 'S'  # Start
    return custom_map

# def get_custom_reward(self, state,custom_reward_regions):
#     # Check if the state falls into any custom reward region
#     for region in custom_reward_regions:
#         row_range, col_range = region['row_range'], region['col_range']
#         if row_range[0] <= state[0] <= row_range[1] and col_range[0] <= state[1] <= col_range[1]:
#             return region['reward']
#
#     # If not in any custom region, return the original reward
#     return reward

def is_inside_triangle(point, vertices, map_size):
    # Check if a point is inside a triangle using barycentric coordinates
    x, y = grid_index_to_tuple(point,map_size)
    x1, y1 = vertices[0]
    x2, y2 = vertices[1]
    x3, y3 = vertices[2]

    detT = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    alpha = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / detT
    beta = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / detT
    gamma = 1 - alpha - beta

    return 0 <= alpha <= 1 and 0 <= beta <= 1 and 0 <= gamma <= 1

def is_behind_obstacle(point, triangle_vertices, obstacle_vertices,map_size):
    # Check if a point is behind a rectangular obstacle

    # Extract the coordinates of the given point (x, y)
    x, y = grid_index_to_tuple(point,map_size)

    # Extract the vertices of the triangular region (visible area)
    x1, y1 = triangle_vertices[0]
    x2, y2 = triangle_vertices[1]
    x3, y3 = triangle_vertices[2]

    # Extract the vertices of the rectangular obstacle
    x_min, y_min = min(obstacle_vertices, key=lambda v: v[0])[0], min(obstacle_vertices, key=lambda v: v[1])[1]
    x_max, y_max = max(obstacle_vertices, key=lambda v: v[0])[0], max(obstacle_vertices, key=lambda v: v[1])[1]

    # Check if the point is behind the rectangular obstacle
    return (x_min <= x <= x_max) and (y_min <= y <= y_max) and is_inside_triangle(point, triangle_vertices, map_size)

def custom_step(self, action, custom_rewards,vertices,map_size, obstacle_vertices,be_sneaky,custom_goal ,state):
    # Call the original step function
    # next_state, reward, done, info = self.original_step(action)
    new_state, reward, done, truncated, info = self.original_step(action)

    #Standard problem :
    # custom_reward = custom_rewards.get(new_state, 0)

    # Modify the reward based on your custom logic
    if be_sneaky:
        if is_inside_triangle(new_state, vertices,map_size):
            if new_state == tuple_to_grid_index(custom_goal,map_size):
                if is_inside_triangle(state,vertices,map_size):
                    custom_reward= -100
                    print("\n")
                    print("Achieve the goal in face of the player")
                    print(custom_reward)
                else :
                    custom_reward = custom_rewards.get(new_state, 0)
            else :
                for obstacle in obstacle_vertices:
                    if is_behind_obstacle(new_state, vertices, obstacle,map_size):
                        if is_behind_obstacle(state, vertices, obstacle, map_size):
                            custom_reward=0
                            break
                        else:
                            custom_reward = 0.1
                            break
                    else:
                        if not is_behind_obstacle(state, vertices, obstacle,map_size):
                            custom_reward = -0.1* 1/(1e-8 +distance_two_point(grid_index_to_tuple(new_state,map_size), custom_goal))
                        else:
                            custom_reward = -0.5 * 1 / (1e-8 + distance_two_point(grid_index_to_tuple(new_state, map_size),custom_goal))
        else:
            #Unseen
            custom_reward = custom_rewards.get(new_state, 0)
    else:
        custom_reward = custom_rewards.get(new_state, 0)
    return new_state, custom_reward, done, truncated, info

# class CustomFrozenLake(gym.Env):
#     def __init__(self, size, hole_locations, goal_location, start_location, custom_rewards, seed):
#         self.size = size
#         self.hole_locations = hole_locations
#         self.goal_location = goal_location
#         self.start_location = start_location
#         self.custom_rewards = custom_rewards
#         self.state = None
#
#         # Define the action and observation spaces
#         self.action_space = gym.spaces.Discrete(4)
#         self.observation_space = gym.spaces.Discrete(size[0] * size[1])
#
#         # Initialize the map
#         self.desc = self.generate_custom_map()
#         self.map_size_flat = self.size[0] * self.size[1]
#
#         # Set the initial state
#         self.reset(seed)
#
#     def generate_custom_map(self):
#         custom_map = np.full((self.size[0], self.size[1]), 'F', dtype='str')  # Initialize with 'H' for holes
#
#         for hole_loc in self.hole_locations:
#             custom_map[hole_loc[0], hole_loc[1]] = 'H'
#
#         custom_map[self.goal_location[0], self.goal_location[1]] = 'G'  # Goal
#         custom_map[self.start_location[0], self.start_location[1]] = 'S'  # Start
#
#         return custom_map
#
#     def reset(self, seed):
#         self.state = self.start_location
#         return self.state
#
#     def step(self, action):
#         # Define the dynamics of the environment based on the action
#         next_state = self.get_next_state(self.state, action)
#
#         # Update the current state
#         self.state = next_state
#
#         # Calculate the reward based on the custom_rewards dictionary
#         reward = self.custom_rewards.get(tuple(next_state), 0)
#
#         # Check if the agent reached the goal or fell into a hole
#         done = (next_state == self.goal_location) or (next_state in self.hole_locations)
#
#         # For simplicity, we use a default observation of the current state
#         observation = next_state
#
#         return next_state,reward,done,done,{}
#         # return observation, reward, done, {}
#
#     def get_next_state(self, current_state, action):
#         # Define the transition dynamics based on the action
#         if action == 0:  # Move Up
#             next_state = (max(current_state[0] - 1, 0), current_state[1])
#         elif action == 1:  # Move Down
#             next_state = (min(current_state[0] + 1, self.size[0] - 1), current_state[1])
#         elif action == 2:  # Move Left
#             next_state = (current_state[0], max(current_state[1] - 1, 0))
#         elif action == 3:  # Move Right
#             next_state = (current_state[0], min(current_state[1] + 1, self.size[1] - 1))
#         else:
#             raise ValueError("Invalid action")
#
#         return next_state


class Qlearning:
    def __init__(self, learning_rate, gamma, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.reset_qtable()

    def update(self, state, action, reward, new_state):
        """Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]"""
        delta = (
            reward
            + self.gamma * np.max(self.qtable[new_state, :])
            - self.qtable[state, action]
        )
        q_update = self.qtable[state, action] + self.learning_rate * delta
        return q_update

    def reset_qtable(self):
        """Reset the Q-table."""
        self.qtable = np.zeros((self.state_size, self.action_size))


class EpsilonGreedy:
    def __init__(self, epsilon, params):
        self.epsilon = epsilon
        self.rng=np.random.default_rng(params.seed)

    def choose_action(self, action_space, state, qtable):
        """Choose an action `a` in the current world state (s)."""
        # First we randomize a number
        explor_exploit_tradeoff = self.rng.uniform(0, 1)

        # Exploration
        if explor_exploit_tradeoff < self.epsilon:
            action = action_space.sample()

        # Exploitation (taking the biggest Q-value for this state)
        else:
            # Break ties randomly
            # If all actions are the same for this state we choose a random one
            # (otherwise `np.argmax()` would always take the first one)
            if np.all(qtable[state, :]) == qtable[state, 0]:
                action = action_space.sample()
            else:
                action = np.argmax(qtable[state, :])
        return action

def run_env(env,params,learner,explorer,custom_rewards,vertices,map_sizes,obstacle_vertices,be_sneaky,custom_goal):
    rewards = np.zeros((params.total_episodes, params.n_runs))
    steps = np.zeros((params.total_episodes, params.n_runs))
    episodes = np.arange(params.total_episodes)
    qtables = np.zeros((params.n_runs, params.state_size, params.action_size))
    all_states = []
    all_actions = []

    for run in range(params.n_runs):  # Run several times to account for stochasticity
        learner.reset_qtable()  # Reset the Q-table between runs

        for episode in tqdm(
            episodes, desc=f"Run {run}/{params.n_runs} - Episodes", leave=False
        ):
            state = env.reset(seed=params.seed)[0]  # Reset the environment
            step = 0
            done = False
            total_rewards = 0

            while not done:
                action = explorer.choose_action(
                    action_space=env.action_space, state=state, qtable=learner.qtable
                )

                # Log all states and actions
                all_states.append(state)
                all_actions.append(action)

                # Take the action (a) and observe the outcome state(s') and reward (r)
                new_state, reward, terminated, truncated, info = env.step(env,action,custom_rewards,vertices,map_sizes,obstacle_vertices,be_sneaky,custom_goal, state)
                done = terminated or truncated

                learner.qtable[state, action] = learner.update(
                    state, action, reward, new_state
                )

                total_rewards += reward
                step += 1

                # Our new state is state
                state = new_state

            # Log all rewards and steps
            rewards[episode, run] = total_rewards
            steps[episode, run] = step
        qtables[run, :, :] = learner.qtable

    return rewards, steps, episodes, qtables, all_states, all_actions

def postprocess(episodes, params, rewards, steps, map_size):
    """Convert the results of the simulation in dataframes."""
    res = pd.DataFrame(
        data={
            "Episodes": np.tile(episodes, reps=params.n_runs),
            "Rewards": rewards.flatten(),
            "Steps": steps.flatten(),
        }
    )
    res["cum_rewards"] = rewards.cumsum(axis=0).flatten(order="F")
    res["map_size"] = np.repeat(f"{map_size}x{map_size}", res.shape[0])

    st = pd.DataFrame(data={"Episodes": episodes, "Steps": steps.mean(axis=1)})
    st["map_size"] = np.repeat(f"{map_size}x{map_size}", st.shape[0])
    return res, st

def qtable_directions_map(qtable, map_size):
    """Get the best learned action & map it to arrows."""
    qtable_val_max = qtable.max(axis=1).reshape(map_size, map_size)
    qtable_best_action = np.argmax(qtable, axis=1).reshape(map_size, map_size)
    directions = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    qtable_directions = np.empty(qtable_best_action.flatten().shape, dtype=str)
    eps = np.finfo(float).eps  # Minimum float number on the machine
    for idx, val in enumerate(qtable_best_action.flatten()):
        if qtable_val_max.flatten()[idx] > eps:
            # Assign an arrow only if a minimal Q-value has been learned as best action
            # otherwise since 0 is a direction, it also gets mapped on the tiles where
            # it didn't actually learn anything
            qtable_directions[idx] = directions[val]
    qtable_directions = qtable_directions.reshape(map_size, map_size)
    return qtable_val_max, qtable_directions

def plot_q_values_map(qtable, env, map_size,params):
    """Plot the last frame of the simulation and the policy learned."""
    qtable_val_max, qtable_directions = qtable_directions_map(qtable, map_size)

    # Plot the last frame
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    ax[0].imshow(env.render())
    ax[0].axis("off")
    ax[0].set_title("Last frame")

    # Plot the policy
    sns.heatmap(
        qtable_val_max,
        annot=qtable_directions,
        fmt="",
        ax=ax[1],
        cmap=sns.color_palette("Blues", as_cmap=True),
        linewidths=0.7,
        linecolor="black",
        xticklabels=[],
        yticklabels=[],
        annot_kws={"fontsize": "xx-large"},
    ).set(title="Learned Q-values\nArrows represent best action")
    for _, spine in ax[1].spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("black")
    img_title = f"frozenlake_q_values_{map_size}x{map_size}.png"
    fig.savefig(params.savefig_folder / img_title, bbox_inches="tight")
    plt.show()

def plot_states_actions_distribution(states, actions, map_size,params):
    """Plot the distributions of states and actions."""
    labels = {"LEFT": 0, "DOWN": 1, "RIGHT": 2, "UP": 3}

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.histplot(data=states, ax=ax[0], kde=True)
    ax[0].set_title("States")
    sns.histplot(data=actions, ax=ax[1])
    ax[1].set_xticks(list(labels.values()), labels=labels.keys())
    ax[1].set_title("Actions")
    fig.tight_layout()
    img_title = f"frozenlake_states_actions_distrib_{map_size}x{map_size}.png"
    fig.savefig(params.savefig_folder / img_title, bbox_inches="tight")
    plt.show()