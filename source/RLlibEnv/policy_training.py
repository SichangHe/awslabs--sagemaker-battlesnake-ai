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
# # Policy Training
#
# This notebook outlines the steps involved in building and deploying a Battlesnake model using Ray RLlib and TensorFlow on Amazon SageMaker.
#
# Library versions currently in use:  TensorFlow 2.1, Ray RLlib 0.8.2
#
# The model is first trained using multi-agent PPO, and then deployed to a managed _TensorFlow Serving_ SageMaker endpoint that can be used for inference.

# %%
import json

import boto3
import botocore
import sagemaker
from sagemaker.rl import RLEstimator, RLToolkit

# %%
with open("../stack_outputs.json") as f:
    info = json.load(f)

# %% [markdown]
# ## Initialise sagemaker
# We need to define several parameters prior to running the training job.

# %%
sm_session = sagemaker.session.Session()
s3_bucket = info["S3Bucket"]

s3_output_path = "s3://{}/".format(s3_bucket)
print("S3 bucket path: {}".format(s3_output_path))

# %%
job_name_prefix = "Battlesnake-job-rllib"

role = info["SageMakerIamRoleArn"]
print(role)

# %% [markdown]
# Change local_mode to True if you want to do local training within this Notebook instance.

# %%
local_mode = False

if local_mode:
    instance_type = "local"
else:
    instance_type = info["SagemakerTrainingInstanceType"]

# If training locally, do some Docker housekeeping..
if local_mode:
    # !/bin/bash ./common/setup.sh
    pass

# %% [markdown]
# # Train your model here

# %%
region = sm_session.boto_region_name
device = "cpu"
image_name = "462105765813.dkr.ecr.{region}.amazonaws.com/sagemaker-rl-ray-container:ray-0.8.2-tf-{device}-py36".format(
    region=region, device=device
)

# %%
# %%time

# Define and execute our training job
# Adjust hyperparameters and train_instance_count accordingly

metric_definitions = [
    {
        "Name": "training_iteration",
        "Regex": "training_iteration: ([-+]?[0-9]*[.]?[0-9]+([eE][-+]?[0-9]+)?)",
    },
    {
        "Name": "episodes_total",
        "Regex": "episodes_total: ([-+]?[0-9]*[.]?[0-9]+([eE][-+]?[0-9]+)?)",
    },
    {
        "Name": "num_steps_trained",
        "Regex": "num_steps_trained: ([-+]?[0-9]*[.]?[0-9]+([eE][-+]?[0-9]+)?)",
    },
    {
        "Name": "timesteps_total",
        "Regex": "timesteps_total: ([-+]?[0-9]*[.]?[0-9]+([eE][-+]?[0-9]+)?)",
    },
    {
        "Name": "training_iteration",
        "Regex": "training_iteration: ([-+]?[0-9]*[.]?[0-9]+([eE][-+]?[0-9]+)?)",
    },
    {
        "Name": "episode_reward_max",
        "Regex": "episode_reward_max: ([-+]?[0-9]*[.]?[0-9]+([eE][-+]?[0-9]+)?)",
    },
    {
        "Name": "episode_reward_mean",
        "Regex": "episode_reward_mean: ([-+]?[0-9]*[.]?[0-9]+([eE][-+]?[0-9]+)?)",
    },
    {
        "Name": "episode_reward_min",
        "Regex": "episode_reward_min: ([-+]?[0-9]*[.]?[0-9]+([eE][-+]?[0-9]+)?)",
    },
    {
        "Name": "episode_len_max",
        "Regex": "episode_len_mean: ([-+]?[0-9]*[.]?[0-9]+([eE][-+]?[0-9]+)?)",
    },
    {
        "Name": "episode_len_mean",
        "Regex": "episode_len_mean: ([-+]?[0-9]*[.]?[0-9]+([eE][-+]?[0-9]+)?)",
    },
    {
        "Name": "episode_len_min",
        "Regex": "episode_len_mean: ([-+]?[0-9]*[.]?[0-9]+([eE][-+]?[0-9]+)?)",
    },
    {
        "Name": "best_snake_episode_len_max",
        "Regex": "best_snake_episode_len_max: ([-+]?[0-9]*[.]?[0-9]+([eE][-+]?[0-9]+)?)",
    },
    {
        "Name": "worst_snake_episode_len_max",
        "Regex": "worst_snake_episode_len_max: ([-+]?[0-9]*[.]?[0-9]+([eE][-+]?[0-9]+)?)",
    },
    {
        "Name": "Snake_hit_wall_max",
        "Regex": "Snake_hit_wall_max: ([-+]?[0-9]*[.]?[0-9]+([eE][-+]?[0-9]+)?)",
    },
    {
        "Name": "Snake_was_eaten_max",
        "Regex": "Snake_was_eaten_max: ([-+]?[0-9]*[.]?[0-9]+([eE][-+]?[0-9]+)?)",
    },
    {
        "Name": "Killed_another_snake_max",
        "Regex": "Killed_another_snake_max: ([-+]?[0-9]*[.]?[0-9]+([eE][-+]?[0-9]+)?)",
    },
    {
        "Name": "Snake_hit_body_max",
        "Regex": "Snake_hit_body_max: ([-+]?[0-9]*[.]?[0-9]+([eE][-+]?[0-9]+)?)",
    },
    {
        "Name": "Starved_max",
        "Regex": "Starved_max: ([-+]?[0-9]*[.]?[0-9]+([eE][-+]?[0-9]+)?)",
    },
    {
        "Name": "Forbidden_move_max",
        "Regex": "Forbidden_move_max: ([-+]?[0-9]*[.]?[0-9]+([eE][-+]?[0-9]+)?)",
    },
]

algorithm = "PPO"
map_size = 11
num_agents = 5
additional_config = {
    "lambda": 0.90,
    "gamma": 0.999,
    "kl_coeff": 0.2,
    "clip_rewards": True,
    "vf_clip_param": 175.0,
    "train_batch_size": 9216,
    "sample_batch_size": 96,
    "sgd_minibatch_size": 256,
    "num_sgd_iter": 3,
    "lr": 5.0e-4,
}

estimator = RLEstimator(
    entry_point="train-mabs.py",
    source_dir="training/training_src",
    dependencies=[
        "training/common/sagemaker_rl",
        "inference/inference_src/",
        "../BattlesnakeGym/",
    ],
    image_uri=image_name,
    role=role,
    train_instance_type=instance_type,
    train_instance_count=1,
    output_path=s3_output_path,
    base_job_name=job_name_prefix,
    metric_definitions=metric_definitions,
    hyperparameters={
        # See train-mabs.py to add additional hyperparameters
        # Also see ray_launcher.py for the rl.training.* hyperparameters
        "num_iters": 10,
        # number of snakes in the gym
        "num_agents": num_agents,
        "iterate_map_size": False,
        "map_size": map_size,
        "algorithm": algorithm,
        "additional_configs": additional_config,
        "use_heuristics_action_masks": False,
    },
)

estimator.fit()

job_name = estimator.latest_training_job.job_name
print("Training job: %s" % job_name)

# %%
# Where is the model stored in S3?
estimator.model_data

# %% [markdown]
# # Create an endpoint to host the policy
# Firstly, we will delete the previous endpoint and model

# %%
sm_client = boto3.client(service_name="sagemaker")
try:
    sm_client.delete_endpoint(EndpointName=info["SagemakerEndPointName"])
    sm_client.delete_endpoint_config(EndpointConfigName=info["SagemakerEndPointName"])
    sm_client.delete_model(ModelName=info["SagemakerEndPointName"])
except botocore.errorfactory.ClientError:
    print("Currently no endpoint")

# %%
# Copy the endpoint to a central location
model_data = "s3://{}/pretrainedmodels/model.tar.gz".format(s3_bucket)
# !aws s3 cp {estimator.model_data} {model_data}

from sagemaker.tensorflow.serving import Model

model = Model(
    model_data=model_data,
    role=role,
    entry_point="inference.py",
    source_dir="inference/inference_src",
    framework_version="2.1.0",
    name=info["SagemakerEndPointName"],
    code_location="s3://{}//code".format(s3_bucket),
)

if local_mode:
    inf_instance_type = "local"
else:
    inf_instance_type = info["SagemakerInferenceInstanceType"]

# Deploy an inference endpoint
predictor = model.deploy(
    initial_instance_count=1,
    instance_type=inf_instance_type,
    endpoint_name=info["SagemakerEndPointName"],
)

# %% [markdown]
# # Test the endpoint
#
# This example is using single observation for a 5-agent environment.
# The last axis is 12 because the current MultiAgentEnv is concatenating 2 frames
# 5 agent maps + 1 food map = 6 maps total    6 maps * 2 frames = 12

from time import time

# %%
import numpy as np

state = np.zeros(shape=(1, 21, 21, 6), dtype=np.float32).tolist()

health_dict = {0: 50, 1: 50}
json = {
    "turn": 4,
    "board": {"height": 11, "width": 11, "food": [], "snakes": []},
    "you": {
        "id": "snake-id-string",
        "name": "Sneky Snek",
        "health": 90,
        "body": [{"x": 1, "y": 3}],
    },
}

before = time()
action_mask = np.array([1, 1, 1, 1]).tolist()

action = predictor.predict(
    {
        "state": state,
        "action_mask": action_mask,
        "prev_action": -1,
        "prev_reward": -1,
        "seq_lens": -1,
        "all_health": health_dict,
        "json": json,
    }
)
elapsed = time() - before

action_to_take = action["outputs"]["heuristics_action"]
print("Action to take {}".format(action_to_take))
print("Inference took %.2f ms" % (elapsed * 1000))

# %% [markdown]
# # Navigation
# - To go back to the introduction click [here](./1_Introduction.ipynb)
# - To build some heuristics click [here](./3_HeuristicsDeveloper.ipynb)

# %%
