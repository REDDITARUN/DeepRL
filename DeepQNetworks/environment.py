# Imports
import cv2
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
from time import time


# Defining the Wumpus World Environment.
class WumpusWorldEnvironment(gym.Env):
    """This class implements the Wumpus World environment."""

    def __init__(self, observation_type, action_type):
        """This method initializes the environment.

        :param string observation_type: - It can take four values: 1. 'integer' 2. 'vector' 3. 'image' 4. 'float'
                                          determining the type of observation returned to the agent.

        :param string action_type: It can take three values: 1. 'discrete' 2. 'continuous' 3. 'multi_discrete'
                                   determining the type of action the agent can take."""

        self.observation_type = observation_type.lower()

        self.environment_width = 6
        self.environment_height = 6

        self.observation_space = spaces.Discrete(self.environment_width * self.environment_height)

        # Action.
        self.action_type = action_type.lower()

        if self.action_type == 'discrete':
            self.action_space = spaces.Discrete(4)
        elif self.action_type == 'continuous':
            self.action_space = spaces.Box(low=np.array([-1]), high=np.array([1]))
        elif self.action_type == 'multi_discrete':
            self.action_space = spaces.MultiDiscrete([2, 2, 2, 2])
        else:
            raise Exception('Invalid action type. Valid action types are: '
                            '\n1. discrete \n2. continuous \n3. multi_discrete')

        # Positions of environment objects.
        self.agent_pos = np.asarray([0, 0])

        self.breeze_pos = np.asarray([[1, 0], [3, 0], [5, 0], [2, 1], [4, 1], [1, 2], [3, 2], [5, 2], [0, 3],
                                      [2, 3], [1, 4], [3, 4], [5, 4], [0, 5], [2, 5], [4, 5]])

        self.gold_pos = np.asarray([4, 5])

        self.pit_pos = np.asarray([[2, 0], [5, 1], [2, 2], [0, 4], [2, 4], [3, 5], [5, 5]])

        self.stench_pos = np.asarray([[3, 2], [2, 3], [4, 3], [3, 4]])

        self.wumpus_pos = np.asarray([3, 3])

        self.gold_quantity = 1

        self.timesteps = 0
        self.max_timesteps = 1000

        # Creating the mapping from the co-ordinates to integers representing the grid blocks.
        self.coordinates_state_mapping = {}
        for i in range(self.environment_height):
            for j in range(self.environment_width):
                self.coordinates_state_mapping[f'{np.asarray([j, i])}'] = i * self.environment_width + j

        self.start_time = time()

    def reset(self):
        """This method resets the agent position and returns the state as the observation.

        :returns observation: - Observation received by the agent (Type depends on the parameter observation_type)."""

        self.agent_pos = np.asarray([0, 0])

        observation = self.return_observation()
        self.timesteps = 0
        self.gold_quantity = 1
        info = {}

        self.start_time = time()

        return observation, info

    def return_observation(self):
        """This method returns the observation.

        :returns observation - Observation received by the agent (Type depends on the parameter observation_type)."""

        if self.observation_type == 'integer':
            observation = self.coordinates_state_mapping[f'{self.agent_pos}']
        elif self.observation_type == 'vector':
            observation = self.agent_pos
        elif self.observation_type == 'image':
            observation = self.render(plot=False)
        elif self.observation_type == 'float':
            time_elapsed = time() - self.start_time
            observation = time_elapsed
        else:
            raise Exception('Invalid observation type. Valid observation types are: '
                            '\n1. integer \n2. vector \n3. image \n4. float')

        return observation

    def take_action(self, action):
        """This method takes the action.

        :param action: - Action taken by the agent (Type depends on the parameter action_type)."""

        if self.action_type == 'discrete':
            if action == 0:
                self.agent_pos[0] += 1  # Right.
            elif action == 1:
                self.agent_pos[0] -= 1  # Left.
            elif action == 2:
                self.agent_pos[1] += 1  # Up.
            elif action == 3:
                self.agent_pos[1] -= 1  # Down.
            else:
                raise Exception('InvalidAction: Discrete action can take values 0, 1, 2 or 3.')

        elif self.action_type == 'continuous':
            if -1 <= action <= -0.5:
                self.agent_pos[0] += 1  # Right.
            elif -0.5 < action <= 0:
                self.agent_pos[0] -= 1  # Left.
            elif 0 < action <= 0.5:
                self.agent_pos[1] += 1  # Up.
            elif 0.5 < action <= 1:
                self.agent_pos[1] -= 1  # Down.
            else:
                raise Exception('InvalidAction: Continuous action has a range [-1, 1].')

        elif self.action_type == 'multi_discrete':
            if action[0] == 1:
                self.agent_pos[0] += 1  # Right.
            if action[1] == 1:
                self.agent_pos[0] -= 1  # Left.
            if action[2] == 1:
                self.agent_pos[1] += 1  # Up.
            if action[3] == 1:
                self.agent_pos[1] -= 1  # Down.
            if len(action) != 4 or (action[0] not in [0, 1] or action[1] not in [0, 1] or action[2] not in [0, 1]
                                    or action[3] not in [0, 1]):
                raise Exception(
                    'InvalidAction: Multi-Discrete action takes binary values in the array [0, 0, 0, 0]. '
                    'Refer to the assignment problem statement on environment details.')

    def step(self, action):
        """This method implements what happens when the agent takes a particular action. It changes the agent's
        position (While not allowing it to go out of the environment space.), maps the environment co-ordinates to a
        state, defines the rewards for the various states, and determines when the episode ends.

        :param action: - Action taken by the agent (Type depends on the parameter action_type).

        :returns observation: - Observation received by the agent (Type depends on the parameter observation_type).
                 int reward: - Integer value that's used to measure the performance of the agent.
                 bool done: - Boolean describing whether the episode has ended.
                 dict info: - A dictionary that can be used to provide additional implementation information."""

        self.take_action(action)

        # Ensuring that the agent doesn't go out of the environment.
        self.agent_pos = np.clip(self.agent_pos, a_min=[0, 0],
                                 a_max=[self.environment_width - 1, self.environment_height - 1])

        observation = self.return_observation()

        reward, terminated, truncated = None, None, None
        info = {}

        return observation, reward, terminated, truncated, info

    def render(self, mode='human', plot=False):
        """This method renders the environment.

        :param string mode: 'human' renders to the current display or terminal and returns nothing.

        :param boolean plot: Boolean indicating whether we show a plot or not.

                             If False, the method returns a resized NumPy array representation of the environment
                             to be used as the state.

                             If True it plots the environment.

        :returns array preprocessed_image: Grayscale NumPy array representation of the environment."""

        fig, ax = plt.subplots(figsize=(15, 15))
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 6)

        def plot_image(plot_pos):
            """This is a helper function to render the environment. It checks which objects are in a particular
            position on the grid and renders the appropriate image.

            :param arr plot_pos: Co-ordinates of the grid position which needs to be rendered."""

            # Initially setting every object to not be plotted.
            plot_agent, plot_breeze, plot_gold, plot_pit, plot_stench, plot_wumpus = \
                False, False, False, False, False, False

            # Checking which objects need to be plotted by comparing their positions.
            if np.array_equal(self.agent_pos, plot_pos):
                plot_agent = True
            if any(np.array_equal(self.breeze_pos[i], plot_pos) for i in range(len(self.breeze_pos))):
                plot_breeze = True
            if self.gold_quantity > 0:  # Gold isn't plotted if it has already been picked by one of the agents.
                if np.array_equal(plot_pos, self.gold_pos):
                    plot_gold = True
            if any(np.array_equal(self.pit_pos[i], plot_pos) for i in range(len(self.pit_pos))):
                plot_pit = True
            if any(np.array_equal(self.stench_pos[i], plot_pos) for i in range(len(self.stench_pos))):
                plot_stench = True
            if np.array_equal(plot_pos, self.wumpus_pos):
                plot_wumpus = True

            # Plot for Agent.
            if plot_agent and \
                    all(not item for item in
                        [plot_breeze, plot_gold, plot_pit, plot_stench, plot_wumpus]):
                agent = AnnotationBbox(OffsetImage(plt.imread('./images/agent.png'), zoom=0.28),
                                       np.add(plot_pos, [0.5, 0.5]), frameon=False)
                ax.add_artist(agent)

            # Plot for Breeze.
            elif plot_breeze and \
                    all(not item for item in
                        [plot_agent, plot_gold, plot_pit, plot_stench, plot_wumpus]):
                breeze = AnnotationBbox(OffsetImage(plt.imread('./images/breeze.png'), zoom=0.28),
                                        np.add(plot_pos, [0.5, 0.5]), frameon=False)
                ax.add_artist(breeze)

            # Plot for Gold.
            elif plot_gold and \
                    all(not item for item in
                        [plot_agent, plot_breeze, plot_pit, plot_stench, plot_wumpus]):
                gold = AnnotationBbox(OffsetImage(plt.imread('./images/gold.png'), zoom=0.28),
                                      np.add(plot_pos, [0.5, 0.5]), frameon=False)
                ax.add_artist(gold)

            # Plot for Pit.
            elif plot_pit and \
                    all(not item for item in
                        [plot_agent, plot_breeze, plot_gold, plot_stench, plot_wumpus]):
                pit = AnnotationBbox(OffsetImage(plt.imread('./images/pit.png'), zoom=0.28),
                                     np.add(plot_pos, [0.5, 0.5]), frameon=False)
                ax.add_artist(pit)

            # Plot for Stench.
            elif plot_stench and \
                    all(not item for item in
                        [plot_agent, plot_breeze, plot_gold, plot_pit, plot_wumpus]):
                stench = AnnotationBbox(OffsetImage(plt.imread('./images/stench.png'), zoom=0.28),
                                        np.add(plot_pos, [0.5, 0.5]), frameon=False)
                ax.add_artist(stench)

            # Plot for Wumpus.
            elif plot_wumpus and \
                    all(not item for item in
                        [plot_agent, plot_breeze, plot_gold, plot_pit, plot_stench]):
                wumpus = AnnotationBbox(OffsetImage(plt.imread('./images/wumpus.png'), zoom=0.28),
                                        np.add(plot_pos, [0.5, 0.5]), frameon=False)
                ax.add_artist(wumpus)

            # Plot for Agent and Breeze.
            elif all(item for item in [plot_agent, plot_breeze]) and \
                    all(not item for item in
                        [plot_gold, plot_pit, plot_stench, plot_wumpus]):
                agent_breeze = AnnotationBbox(OffsetImage(plt.imread('./images/agent_breeze.png'), zoom=0.28),
                                              np.add(plot_pos, [0.5, 0.5]), frameon=False)
                ax.add_artist(agent_breeze)

            # Plot for Agent and Pit.
            elif all(item for item in [plot_agent, plot_pit]) and \
                    all(not item for item in
                        [plot_breeze, plot_gold, plot_stench, plot_wumpus]):
                agent_pit = AnnotationBbox(OffsetImage(plt.imread('./images/agent_dead_pit.png'), zoom=0.28),
                                           np.add(plot_pos, [0.5, 0.5]), frameon=False)
                ax.add_artist(agent_pit)

            # Plot for Agent and Stench.
            elif all(item for item in [plot_agent, plot_stench]) and \
                    all(not item for item in
                        [plot_breeze, plot_gold, plot_pit, plot_wumpus]):
                agent_stench = AnnotationBbox(OffsetImage(plt.imread('./images/agent_stench.png'), zoom=0.28),
                                              np.add(plot_pos, [0.5, 0.5]), frameon=False)
                ax.add_artist(agent_stench)

            # Plot for Agent, Breeze and Stench.
            elif all(item for item in [plot_agent, plot_breeze, plot_stench]) and \
                    all(not item for item in
                        [plot_gold, plot_pit, plot_wumpus]):
                agent_breeze_stench = AnnotationBbox(OffsetImage(plt.imread('./images/agent_breeze_stench.png'),
                                                                 zoom=0.28), np.add(plot_pos, [0.5, 0.5]),
                                                     frameon=False)
                ax.add_artist(agent_breeze_stench)

            # Plot for Agent and Wumpus.
            elif all(item for item in [plot_agent, plot_wumpus]) and \
                    all(not item for item in
                        [plot_gold, plot_pit, plot_stench, plot_breeze]):
                agent_wumpus = AnnotationBbox(OffsetImage(plt.imread('./images/agent_dead_wumpus_alive.png'),
                                                          zoom=0.28), np.add(plot_pos, [0.5, 0.5]), frameon=False)
                ax.add_artist(agent_wumpus)

            # Plot for Breeze and Gold.
            elif all(item for item in [plot_breeze, plot_gold]) and \
                    all(not item for item in
                        [plot_agent, plot_pit, plot_stench, plot_wumpus]):
                breeze_gold = AnnotationBbox(OffsetImage(plt.imread('./images/breeze_gold.png'), zoom=0.28),
                                             np.add(plot_pos, [0.5, 0.5]), frameon=False)
                ax.add_artist(breeze_gold)

            # Plot for Breeze and Stench.
            elif all(item for item in [plot_breeze, plot_stench]) and \
                    all(not item for item in
                        [plot_agent, plot_gold, plot_pit, plot_wumpus]):
                breeze_stench = AnnotationBbox(OffsetImage(plt.imread('./images/breeze_stench.png'), zoom=0.28),
                                               np.add(plot_pos, [0.5, 0.5]), frameon=False)
                ax.add_artist(breeze_stench)

            # Plot for Breeze, Stench, and Gold.
            elif all(item for item in [plot_breeze, plot_gold, plot_stench]) and \
                    all(not item for item in
                        [plot_agent, plot_pit, plot_wumpus]):
                breeze_gold_stench = AnnotationBbox(OffsetImage(plt.imread('./images/breeze_gold_stench.png'),
                                                                zoom=0.28), np.add(plot_pos, [0.5, 0.5]),
                                                    frameon=False)
                ax.add_artist(breeze_gold_stench)

            # Plot for Stench and Gold.
            elif all(item for item in [plot_stench, plot_gold]) and \
                    all(not item for item in
                        [plot_agent, plot_breeze, plot_pit, plot_wumpus]):
                stench_gold = AnnotationBbox(OffsetImage(plt.imread('./images/stench_gold.png'), zoom=0.28),
                                             np.add(plot_pos, [0.5, 0.5]), frameon=False)
                ax.add_artist(stench_gold)

        coordinates_state_mapping_2 = {}
        for j in range(self.environment_height * self.environment_width):
            coordinates_state_mapping_2[j] = np.asarray(
                [j % self.environment_width, int(np.floor(j / self.environment_width))])

        # Rendering the images for all states.
        for position in coordinates_state_mapping_2:
            plot_image(coordinates_state_mapping_2[position])

        plt.xticks([0, 1, 2, 3, 4, 5])
        plt.yticks([0, 1, 2, 3, 4, 5])
        plt.grid()

        if plot:  # Displaying the plot.
            plt.show()
        else:  # Returning the preprocessed image representation of the environment.
            fig.canvas.draw()
            img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :1]
            width = int(84)
            height = int(84)
            dim = (width, height)
            # noinspection PyUnresolvedReferences
            preprocessed_image = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            plt.close(fig)
            return preprocessed_image
