import os
import random
import sys

import numpy as np
from stable_baselines3 import PPO

cwd = '/kaggle_simulations/agent/'
if os.path.exists(cwd):
  sys.path.append(cwd)
else:
  cwd = ''

pmodel = PPO.load(f'{cwd}model.params')

def act(observation, configuration):
    # Use the best model to select a column
    col, _ = pmodel.predict(np.array(observation['board']).reshape(1, 6,7))
    # Check if selected column is valid
    is_valid = (observation['board'][int(col)] == 0)
    # If not valid, select random move.
    if is_valid:
        return int(col)
    else:
        return random.choice([col for col in range(configuration.columns) if observation.board[int(col)] == 0])