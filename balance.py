import math
import matplotlib.pyplot as plt
import gym
import numpy as np
from collections import defaultdict
import abc
import random

PRINT = False


class Learner:
    def __init__(self, learning_rate=0.4, discount=0.98, exploration_rate=0.4):
        self.environment = gym.make('CartPole-v1')
        self.attempt_no = 1
        self.upper_bounds = [
            self.environment.observation_space.high[0],
            0.5,
            self.environment.observation_space.high[2],
            math.radians(50)
        ]

        self.lower_bounds = [
            self.environment.observation_space.low[0],
            -0.5,
            self.environment.observation_space.low[2],
            -math.radians(50)
        ]
        self.n_actions = 2
        self.lr = learning_rate
        self.discount = discount
        self.exploration_rate = exploration_rate
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions)) 


        self.set_buckets_percentages([
            [0.25, 0.125, 0.1, 0.05, 0.1, 0.125, 0.25], # Cart Position percentage splits
            [0.35, 0.15, 0.15, 0.35], # Cart Velocity percentage splits
            [0.25, 0.125, 0.1, 0.05, 0.1, 0.125, 0.25],       # Pole Angle percentage splits
            [0.3, 0.15, 0.05, 0.05, 0.15, 0.3]   # Pole Angular Velocity percentage splits
        ])

        self.learning = True
        self.display = False



    def set_buckets_percentages(self, percentages):
        self.percentages = percentages

        for b in self.percentages:
            for i in range(1, len(b)):
                b[i] += b[i-1]

        self.buckets = self.calculate_bucket_boundaries()

    
    def calculate_bucket_boundaries(self):
        """Calculate boundaries for each dimension based on percentage splits."""
        boundaries = []
        for i, percent_splits in enumerate(self.percentages):
            lower = self.lower_bounds[i]
            upper = self.upper_bounds[i]
            range_size = upper - lower
            boundaries.append([lower + p * range_size for p in percent_splits])
        return boundaries

    def learn(self, max_attempts):
        rewards = []
        self.learning = True
        for _ in range(max_attempts):
            reward_sum = self.attempt()

            rewards.append(reward_sum)
            if reward_sum > 50 and PRINT:
                print(f"Attempt: {self.attempt_no}, Reward Sum: {reward_sum}")

        return rewards

    @abc.abstractmethod
    def attempt(self) -> float:
        return 0

    
    def discretise(self, observation):
        """Convert continuous state into a discrete one by assigning buckets with non-uniform lengths."""
        discretised_observation = []
        for i in range(len(observation)):
            value = observation[i]
            # Assign bucket index based on custom boundaries for each dimension
            bucket_index = np.digitize(value, self.buckets[i], right=True)
            discretised_observation.append(bucket_index)
        return tuple(discretised_observation)  # Tuple as dictionary key for Q-table


    def pick_action(self, observation):
        """Epsilon-greedy action selection."""
        expl = self.exploration_rate
        if self.learning and random.random() < expl:
            return self.environment.action_space.sample()  # Exploration
        else:
            return int(np.argmax(self.q_table[observation]))  # Exploitation

    def run_display(self):
        prev = self.display
        self.display = True
        self.attempt()
        self.display = prev


class QLearner(Learner):
    def attempt(self):
        observation = self.discretise(self.environment.reset())  # Discretize the initial observation
        done = False
        reward_sum = 0.0
        while not done:
            if self.display:
                self.environment.render()
            action = self.pick_action(observation)  # Choose action based on current state
            new_observation, reward, done, info = self.environment.step(action)  # Take action
            new_observation = self.discretise(new_observation)
            self.update_knowledge(action, observation, new_observation, reward)  # Update Q-table
            observation = new_observation
            reward_sum += reward
        self.attempt_no += 1
        return reward_sum

    def update_knowledge(self, action, observation, new_observation, reward):
        if not self.learning: return
        """Q-learning update rule."""
        best_future_q = np.max(self.q_table[new_observation])  # Max Q-value for the next state
        current_q = self.q_table[observation][action]
        # Q-learning formula
        self.q_table[observation][action] = (1 - self.lr) * current_q + self.lr * (reward + self.discount * best_future_q)



class SarsaLearner(Learner):
    def attempt(self):
        observation = self.discretise(self.environment.reset())  # Discretize the initial observation
        action = self.pick_action(observation)  # Choose action based on current state
        done = False
        reward_sum = 0.0
        while not done:
            if self.display:
                self.environment.render()
            new_observation, reward, done, info = self.environment.step(action)  # Take action
            new_observation = self.discretise(new_observation)
            next_action = self.pick_action(new_observation)
            self.update_knowledge(action, next_action, observation, new_observation, reward)  # Update Q-table
            observation = new_observation
            action = next_action
            reward_sum += reward
        self.attempt_no += 1
        return reward_sum


    def update_knowledge(self, action, next_action, observation, new_observation, reward):
        if not self.learning: return
        current_q = self.q_table[observation][action]
        next_q = self.q_table[new_observation][next_action]

        self.q_table[observation][action] = current_q + self.lr * (reward + self.discount*next_q - current_q)





class LLLearner:
    def __init__(self, learning_rate=0.4, discount=0.98, exploration_rate=0.4):
        self.environment = gym.make("LunarLander-v2")
        self.attempt_no = 1
        self.upper_bounds = self.environment.observation_space.high
        self.lower_bounds = self.environment.observation_space.low

        self.n_actions = 4
        self.lr = learning_rate
        self.discount = discount
        self.exploration_rate = exploration_rate
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions))  # Q-table initialized to zero


        self.set_buckets_percentages([
            [0.25, 0.15, 0.09, 0.02, 0.09, 0.15, 0.25], # X position
            list(reversed([0.20, 0.15, 0.10, 0.10, 0.05, 0.05, 0.10, 0.20])),                   # y position
            [], # x linear velocity
            [],         # y linear velocity
            [0.3, 0.15, 0.04, 0.02, 0.04, 0.15, 0.3],         # angle
            [0.3, 0.15, 0.04, 0.02, 0.04, 0.15, 0.3],         # angular velocity
            [],         # left leg contact
            []          # right leg contact
        ])

        self.buckets[2] = [-2.0, -1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0, 2.0, self.upper_bounds[2]]
        self.buckets[3] = [-2.0, -1.0, -0.5, -0.2, 0.0, 1, self.upper_bounds[3]]

        self.learning = True
        self.display = False



    def set_buckets_percentages(self, percentages):
        self.percentages = percentages

        for b in self.percentages:
            for i in range(1, len(b)):
                b[i] += b[i-1]

        self.buckets = self.calculate_bucket_boundaries()

    
    def calculate_bucket_boundaries(self):
        """Calculate boundaries for each dimension based on percentage splits."""
        boundaries = []
        for i, percent_splits in enumerate(self.percentages):
            lower = self.lower_bounds[i]
            upper = self.upper_bounds[i]
            range_size = upper - lower
            boundaries.append([lower + p * range_size for p in percent_splits])
        return boundaries

    def learn(self, max_attempts):
        rewards = []
        self.learning = True
        for _ in range(max_attempts):
            reward_sum = self.attempt()
            rewards.append(reward_sum)
            if reward_sum > 50 and PRINT:
                print(f"Attempt: {self.attempt_no}, Reward Sum: {reward_sum}")

        return rewards

    @abc.abstractmethod
    def attempt(self) -> float:
        return 0

    
    def discretise(self, observation):
        """Convert continuous state into a discrete one by assigning buckets with non-uniform lengths."""
        observation = list(observation)
        discretised_observation = []
        for i in range(len(observation) - 2):
            value = observation[i]
            # Assign bucket index based on custom boundaries for each dimension
            bucket_index = np.digitize(value, self.buckets[i], right=True)
            discretised_observation.append(bucket_index)
        return tuple(discretised_observation + observation[-2:])  # Tuple as dictionary key for Q-table


    def pick_action(self, observation):
        """Epsilon-greedy action selection."""
        expl = self.exploration_rate
        if self.learning and random.random() < expl:
            return self.environment.action_space.sample()  # Exploration
        else:
            return int(np.argmax(self.q_table[observation]))  # Exploitation

    def run_display(self):
        prev = self.display
        self.display = True
        self.attempt()
        self.display = prev


class LLQLearner(LLLearner):
    def attempt(self):
        observation = self.discretise(self.environment.reset())  # Discretize the initial observation
        done = False
        reward_sum = 0.0
        while not done:
            if self.display:
                self.environment.render()
            action = self.pick_action(observation)  # Choose action based on current state
            new_observation, reward, done, info = self.environment.step(action)  # Take action
            new_observation = self.discretise(new_observation)
            self.update_knowledge(action, observation, new_observation, reward)  # Update Q-table
            observation = new_observation
            reward_sum += reward
        self.attempt_no += 1
        return reward_sum

    def update_knowledge(self, action, observation, new_observation, reward):
        """Q-learning update rule."""
        best_future_q = np.max(self.q_table[new_observation])  # Max Q-value for the next state
        current_q = self.q_table[observation][action]
        # Q-learning formula
        self.q_table[observation][action] = (1 - self.lr) * current_q + self.lr * (reward + self.discount * best_future_q)






def main():
    global PRINT
    PRINT = True
    learner = QLearner()
    rewards = learner.learn(10000)

    plt.plot(rewards)
    plt.show()



if __name__ == '__main__':
    main()
