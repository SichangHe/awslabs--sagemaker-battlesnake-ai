# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (TensorFlow JumpStart)
#     language: python
#     name: HUB_1P_IMAGE
# ---

# %% [markdown]
# **Note**: When running this notebook on SageMaker Studio, you should make sure the 'SageMaker JumpStart Tensorflow 1.0' image/kernel is used. You can run run all cells at once or step through the notebook.
# # Reinforcement Learning for Battlesnake AI Competitions Revision 1.1
#
# This project shows how to build and deploy an AI for the platform [Battlesnake](https://play.battlesnake.com/) on AWS with [Amazon Sagemaker](https://aws.amazon.com/sagemaker/)!
#
# It is ready to deploy and contains learning materials for AI enthusiasts.
#
# __What is Battlesnake?__ (taken from [battlesnake.com](https://docs.battlesnake.com/references/rules)):
#
# > Battlesnake is an autonomous survival game where your snake competes with others to find and eat food without being eliminated. To accomplish this, you will have to teach your snake to navigate the serpentine paths created by walls, other snakes, and their own growing tail without running out of energy.
#
# ## Introduction
#
# This project contains a ready-to-use AI for Battlesnake as well as a development environment that can be used to modify and improve the AI.
# The included AI makes movement decisions in two steps:
#
#   1. [__Train a Neural Network Policy__](./2_PolicyTraining.ipynb)
#   2. [__Run Heuristics__](./3_HeuristicsDeveloper.ipynb)
#
# Several pre-trained neural network models are provided within this project as well as some default heuristics. These pre-trained models (snakes) are not designed to win the Battlesnake competition, so you'll have to improve them in order to have a chance of winning. The training environment is provided for you to make modifications and to retrain the neural network models in order to obtain better results.
#
# The heuristics module allows you to provide additional code that can override the neural network's predicted best action so that your snake avoids colliding with walls, eats food if it is safe to do so, attempts to eat a competitor snake, ...
#
# ### Architecture
#
# If you use Steps 1-3, you will have the following deployed within your AWS account:
#
# ![General Architecture](https://github.com/awslabs/sagemaker-battlesnake-ai/blob/master/Documentation/images/ArchitectureSagemakerBattlesnakeFull.png?raw=true "General Architecture")
#
# ### Testing your snake
#
# Head to https://play.battlesnake.com/ and [create your own snake](https://play.battlesnake.com/account/snakes/create/).
# Enter the your snake's name and in the `URL` field, enter the outputs of the following cell:

# %%
import json

with open("../stack_outputs.json") as f:
    info = json.load(f)
print("Your Snake URL is: {}".format(info["SnakeAPI"]))

# %% [markdown]
# ## Training an RL snake
#
# Open the [PolicyTraining.ipynb](./2_PolicyTraining.ipynb) notebook and read through the steps. Press ► at the top of the Jupyter window to run the notebook.
#
# ## Open the training source code
#
# Open [`train-mabs.py`](./training/training_src/train-mabs.py), here you can edit the algorithms, hyperparameters, and other configurations to train your own policy.
#
# See https://docs.ray.io/en/master/rllib.html for more details on how to improve your policy.
#
# ## How to develop your own heuristic algorithms
#
# Open the notebook [`HeuristicDeveloper.ipynb`](./3_HeuristicsDeveloper.ipynb) and ensure that you have a functioning policy (if you have altered the model, you may need to configure the inference step in [`heuristics_utils.get_action(*args)`](./heuristics_utils.py)).
#
# ### Open the heuristic source code
#
# Navigate to [`RLlibEnv/inference/inference_src/battlesnake_heuristics.py`](./inference/inference_src/battlesnake_heuristics.py)
#
# You can customize the `run()` method in the class `MyBattlesnakeHeuristics` with your own rules (see `go_to_food_if_close` for an example).
#
# ## Visualizing your algorithm
#
# - If you want to visualize your AI in action, ensure that you are using *Jupyter* instead of *JupyterLab* (this is the default if you use the links from the CloudFormation 'Outputs' tab).
# - The notebook loads a pre-trained model and allows your AI to interact with the environment
# - After the *Playback the simulation* section, you should see the step-by-step positions, actions, health etc. of each snake.
# - If you want to specify the positions of each snake and food (instead of randomly generating it), you can enter it in `initial_state` in *Define the openAI gym*. initial_state is defined similarly to the [battlesnake API](https://docs.battlesnake.com/snake-api).
#
# ![Visualization](https://github.com/awslabs/sagemaker-battlesnake-ai/blob/master/Documentation/images/VisualizingHeuristics.png?raw=true "Visualize the heuristics")
#
# # Navigation
# - To train a new model click [here](./2_PolicyTraining.ipynb)
# - To build some heuristics click [here](./3_HeuristicsDeveloper.ipynb)

# %%
