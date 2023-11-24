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
    """Transform a tuple (x,y) to the related index in the grid"""
    return tuple[0]*map_size +tuple[1]

def grid_index_to_tuple(index,map_size):
    """Transform teh index grix to a tuple (x,y)"""
    return index//map_size,index%map_size
def distance_two_point(tuple1,tuple2):
    """Give the distance between to points designed by their tuple"""
    return np.linalg.norm(np.array(tuple1) - np.array(tuple2))/10

def create_custom_frozenlake(size, hole_locations, goal_location, start_location):
    """Create a map with predefined holes, goal and starting location"""
    custom_map = generate_custom_map(size, hole_locations,goal_location, start_location)
    env = gym.make('FrozenLake-v1', desc=custom_map, is_slippery=False, render_mode="rgb_array")
    return env


def generate_custom_map(size, hole_locations, goal_location, start_location):
    """Design the map by placing holes, goal and starting point"""
    custom_map = np.full((size[0], size[1]), 'F', dtype='str')  # Initialize with 'H' for holes

    for hole_loc in hole_locations:
        custom_map[hole_loc[0], hole_loc[1]] = 'H'

    custom_map[goal_location[0], goal_location[1]] = 'G'  # Goal
    custom_map[start_location[0], start_location[1]] = 'S'  # Start
    return custom_map

def is_inside_triangle(point, vertices, map_size):
    """Given the vertices of the visible triangle, return if the agent is inside it or not"""
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
    """Given the visible triangle and obstacles, check if the agent is hidden or directly visible """
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

def custom_step(self, action, custom_rewards,vertices,map_size, obstacle_vertices,be_sneaky,custom_goal ,state
                ,custom_holes, nb_be_hidden):
    """Define a specific step so as to design the reward function to bypass an ennemy in a credible way"""
    # Call the original step function
    new_state, reward, done, truncated, info = self.original_step(action)

    # Modify the reward based on your custom logic
    ###If we want to do it in a credible way
    if be_sneaky:
        if grid_index_to_tuple(new_state,map_size) in custom_holes:
            done = True  # Episode is done if the agent moves into a hole
            custom_reward = -2  # Assign a negative reward for falling into a hole
        else:
            if is_inside_triangle(state, vertices,map_size) and new_state == tuple_to_grid_index(custom_goal,map_size): #If the agent was directly seen and will converge to the goal
                custom_reward= -1000
                print("Achieve the goal facing the player")
                print(custom_reward)
            if is_inside_triangle(new_state, vertices,map_size):
                if new_state == tuple_to_grid_index(custom_goal,map_size):
                    if is_inside_triangle(state,vertices,map_size):
                        custom_reward= -1000
                        print("Achieve the goal facing the player")
                        print(custom_reward)
                    else :
                        custom_reward = custom_rewards.get(new_state, 0) #If the agent reachs the goal while not being in the visible triangular
                else :
                    for obstacle in obstacle_vertices:
                        if is_behind_obstacle(new_state, vertices, obstacle,map_size):
                            if is_behind_obstacle(state, vertices, obstacle, map_size):
                                custom_reward= 1 - 1.1**nb_be_hidden #Get a reward for staying in a hidden area and penalizes staying forever in there
                                nb_be_hidden+=1
                                break
                            else:
                                custom_reward = 0 #Get a reward for finding a hidden area
                                nb_be_hidden = 0
                                break
                        else:
                            if not is_behind_obstacle(state, vertices, obstacle,map_size): #Penalizes staying in a visible area
                                custom_reward = -0.1* 1/(1e-8 +distance_two_point(grid_index_to_tuple(new_state,map_size), custom_goal))
                            else: #Penalizes more becoming visible
                                custom_reward = -0.5 * 1 / (1e-8 + distance_two_point(grid_index_to_tuple(new_state, map_size),custom_goal))
            else:
                #If the agent is not in the visible triangle
                custom_reward = custom_rewards.get(new_state, 0)
    else: #Standard behavior : benchmark
        custom_reward = custom_rewards.get(new_state, 0)
    return new_state, custom_reward, done, truncated, info, nb_be_hidden

class Qlearning:
    """Perform the Q-learning step"""
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
    """Choose an action while being epsilon greedy"""
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
    

def run_env(env,params,learner,explorer,custom_rewards,vertices,map_sizes,obstacle_vertices,be_sneaky,custom_goal,custom_holes):
    """Main function that plays the game"""
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
            nb_be_hidden = 0

            while not done:
                action = explorer.choose_action(
                    action_space=env.action_space, state=state, qtable=learner.qtable
                )

                # Log all states and actions
                all_states.append(state)
                all_actions.append(action)

                # Take the action (a) and observe the outcome state(s') and reward (r)
                new_state, reward, terminated, truncated, info, nb_be_hidden = env.step(env,action,custom_rewards,vertices,map_sizes,
                                                                          obstacle_vertices,be_sneaky,custom_goal,
                                                                          state,custom_holes,nb_be_hidden)
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

def qtable_directions_map(qtable, map_size, custom_holes):
    """Get the best learned action & map it to arrows."""
    qtable_val_max = qtable.max(axis=1).reshape(map_size, map_size)
    qtable_best_action = np.argmax(qtable, axis=1).reshape(map_size, map_size)
    directions = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    qtable_directions = np.empty(qtable_best_action.flatten().shape, dtype=str)

    # Normalize Q-values to the range [0, 1]
    qtable_normalized = (qtable - qtable.min()) / (qtable.max() - qtable.min())

    eps = np.finfo(float).eps  # Minimum float number on the machine
    for idx, val in enumerate(qtable_best_action.flatten()):
        if qtable_normalized.flatten()[idx] > eps:
            if grid_index_to_tuple(idx,map_size) in custom_holes:
                pass
            else:
                # Assign an arrow only if a minimal Q-value has been learned as best action
                # otherwise since 0 is a direction, it also gets mapped on the tiles where
                # it didn't actually learn anything
                qtable_directions[idx] = directions[val]
    qtable_directions = qtable_directions.reshape(map_size, map_size)
    return qtable_val_max, qtable_directions

def plot_q_values_map(qtable, env, map_size,params, custom_holes):
    """Plot the last frame of the simulation and the policy learned."""
    qtable_val_max, qtable_directions = qtable_directions_map(qtable, map_size, custom_holes)

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