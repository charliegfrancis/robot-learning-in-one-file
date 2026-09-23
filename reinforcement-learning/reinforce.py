import argparse
import os

import torch
import gymnasium as gym
from skrl.utils import set_seed
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
import datetime
from torch.utils.tensorboard import SummaryWriter

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default=None, help="Load checkpoint from path")
parser.add_argument("--eval", type=bool, action="store_true", help="Run in evaluation mode (no dropout or batchnorm)")

# create the agent class, inheriting from torch.nn.Module
class agent(torch.nn.Module):
    state_size = 0
    action_size = 0
    def __init__(self, state_size, action_size):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size

        # define actor network - output is a vector double the size of the action space (see line 45)
        self.actor_network = torch.nn.Sequential(
            torch.nn.Linear(self.state_size, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 64),
            torch.nn.ELU(),
            torch.nn.Linear(64, self.action_size*2),
        )

        # create optimiser with self.parameters()
        self.optim = torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, state):
        logits = self.actor_network(state)
        return logits

    def act(self, state):
        logits = self.forward(state)   # run forward pass
        means, stds = torch.split(logits, self.action_size, dim=1)  # treat first half of actor output vector as the means, second half as the stds
        stds = torch.nn.functional.softplus(stds) + 1e-3
        dist = torch.distributions.Normal(loc=means, scale=stds)   # sample from output action distribution (continuous action space)
        actions = dist.sample()
        logprobs = dist.log_prob(actions).sum(dim=-1)
        return actions.detach().clamp(-1.0, 1.0), logprobs

    def backward(self, loss):
        self.optim.zero_grad()  # Clear old gradients
        loss.backward()         # Calculate new gradients
        self.optim.step()       # Update weights

# get the environment from isaaclab
task_name = "Isaac-Cartpole-Direct-v0"
num_envs = 1024
env = load_isaaclab_env(task_name=task_name, parser=parser, num_envs=num_envs)
env = wrap_env(env)
device = env.device

# defer parsing of arguments to include loader arguments (run with --help to see all the arguments)
args, _ = parser.parse_known_args()

# seed for reproducibility
set_seed(42)

# hyperparams
n_rollouts = 2000
max_timesteps = 500
gamma = 0.98

# create instance of agent
model = agent(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
if args.checkpoint:
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
else:
    model = agent(env.observation_space.shape[0], env.action_space.shape[0]).to(device)

if args.eval:
    model.eval()

def train():

    # run a rollout - a full episode of the agent attempting the task
    # this actually runs num_envs rollouts each iteration, but pytorch handles the multiple envs so the code is the same as for one
    for n in range(n_rollouts):
        # reset rewards, log probabilities, done for each rollout
        observation, info = env.reset()
        rewards = []
        probs = []
        dones = []

        # run the rollout up to max timesteps
        for t in range(max_timesteps):
            action, prob = model.act(observation)
            probs.append(prob) # collect log_probs of each action, this is how gradients will flow into the network
            observation, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)
            dones.append(torch.logical_or(terminated, truncated))
        steps = len(rewards)

        # this is full monte-carlo, so we back-calculate actual returns for the episode (no bootstrapping)
        returns = steps*[0]
        for i in range(steps-1, -1, -1):
            returns[i] = rewards[i]
            if(i<steps-1):
                returns[i] = rewards[i] + gamma * returns[i+1] * (~dones[i]).float()    # this is a montecarlo application of the bellman equation
                # note we run multiple envs in parallel, hence why we have to zero out the envs where the episode ended.
                # new eps will start and get unfairly truncated, but I think this way is easier to understand than the single total timesteps loop

        # normalise the returns and prepare the tensors
        returns = torch.stack(returns).squeeze()
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)   # normalise returns

        # calculate loss - here the computation graph must be maintained, so that the optimiser can update the weights with respect to the loss (via the log probs)
        policy_loss = []
        for i in range(steps):
            policy_loss.append(-1 * probs[i] * returns[i])
        total_loss = torch.stack(policy_loss).sum(dim=0).mean()  # chosen to sum losses over each trajectory then average over parallel envs (to keep the loss magnitude same as for 1 env)

        # backpropagate and update weights
        model.backward(total_loss)

        # log mean total rewards per rollout to tensorboard
        print(f"rollout {n*num_envs} / {n_rollouts*num_envs} : actor loss = {total_loss.item()} : reward = {torch.stack(rewards).sum(dim=0).mean().item()}")
        writer.add_scalar("Reward / Total reward (mean)", torch.stack(rewards).sum(dim=0).mean().item(), n*num_envs)
        writer.add_scalar("loss/policy", total_loss.item(), n*num_envs)

        # save the model weights locally
        torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), "reinforce_cartpole.pth"))

# set up tensorboard log session
run_name = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f") + "_REINFORCE"
writer = SummaryWriter(log_dir=f"runs/torch/{task_name}/{run_name}")

# train model
train()