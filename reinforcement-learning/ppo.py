import argparse

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
parser.add_argument("--eval", action="store_true", help="Run in evaluation mode (no dropout or batchnorm)")

# create the agent class, inheriting from torch.nn.Module
class agent(torch.nn.Module):
    state_size = 0
    action_size = 0
    def __init__(self, state_size, action_size):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size

        # define actor network - output is a vector double the size of the action space (see line 57)
        self.actor_network = torch.nn.Sequential(
            torch.nn.Linear(self.state_size, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 64),
            torch.nn.ELU(),
            torch.nn.Linear(64, self.action_size*2),
        )

        # define value network - output is just a single number - the estimated value of the state
        self.critic_network = torch.nn.Sequential(
            torch.nn.Linear(self.state_size, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 64),
            torch.nn.ELU(),
            torch.nn.Linear(64, 1),
        )

        # create optimiser with self.parameters()
        self.optim = torch.optim.Adam(self.parameters(), lr=5e-4)

    def forward(self, state):
        logits = self.actor_network(state)
        value_logits = self.critic_network(state)    # when we have a new state we need a new value estimate as well
        return logits, value_logits

    def act(self, state):
        logits, value_logits = self.forward(state)   # run forward pass
        means, stds = torch.split(logits, self.action_size, dim=1)  # treat first half of actor output vector as the means, second half as the stds
        stds = torch.nn.functional.softplus(stds) + 1e-3
        dist = torch.distributions.Normal(loc=means, scale=stds)   # sample from output action distribution (continuous action space)
        actions = dist.sample()
        logprobs = dist.log_prob(actions).sum(dim=-1)
        return actions.detach(), logprobs, value_logits

    # used to re-calculate how likely the model now is to select an old state-action pair (central to how PPO works - see below)
    def evaluate_new_prob(self, old_state, old_action):
        logits, value_logits = self.forward(old_state)   # run forward pass
        means, stds = torch.split(logits, self.action_size, dim=-1)  # treat first half of actor output vector as the means, second half as the stds
        stds = torch.nn.functional.softplus(stds) + 1e-3
        dist = torch.distributions.Normal(loc=means, scale=stds)   # sample from output action distribution (continuous action space)
        new_logprobs = dist.log_prob(old_action).sum(dim=-1)
        return new_logprobs, value_logits

    def backward(self, loss):
        self.optim.zero_grad()  # Clear old gradients
        loss.backward()         # Calculate new gradients
        torch.nn.utils.clip_grad_norm_(self.actor_network.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(), 1.0)
        self.optim.step()       # Update weights

# get the environment from isaaclab
task_name = "Isaac-Ant-Direct-v0"
num_envs = 512   # Isaac is designed for massively parallel envs; a single env gives very noisy gradients (lower this if you run out of GPU memory)
env = load_isaaclab_env(task_name=task_name, parser=parser, num_envs=num_envs)
env = wrap_env(env)
device = env.device

# defer parsing of arguments to include loader arguments (run with --help to see all the arguments)
args, _ = parser.parse_known_args()

# seed for reproducibility
set_seed(42)

# hyperparams
n_rollouts = 2000
max_timesteps = 999
gamma = 0.98
value_loss_coeff = 0.5
ppo_num_training_steps_per_epoch = 10
ppo_clip_size = 0.2

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
        # reset rewards, log probabilities, values, done for each rollout
        observation, info = env.reset()
        rewards = []
        observations = []
        actions = []
        probs = []
        values = []
        dones = []

        # run the rollout up to max timesteps
        for t in range(max_timesteps):
            with torch.no_grad():   # no grads needed while collecting; we re-evaluate during the update
                action, prob, value = model.act(observation)
            observations.append(observation)
            actions.append(action)
            probs.append(prob) # collect log_probs of each action, this is how gradients will flow into the network
            values.append(value)
            observation, reward, terminated, truncated, info = env.step(action.clamp(-1.0, 1.0))
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

        # prepare the tensors (keep raw returns - they are the critic's regression target)
        returns = torch.stack(returns).squeeze()
        values = torch.stack(values).squeeze()  # turn a list of 1d tensors into a single tensor

        # calculate the advantage as per A2C - the difference between the return the policy acheived and the estimated value of the state
        advantages = returns - values.detach()  # using detach so that the critic network gradients are not updated by the actor backpropagation
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)   # normalise advantages (NOT returns) for a stable policy signal

        old_probs = torch.stack(probs).detach()
        observations = torch.stack(observations)
        actions = torch.stack(actions)

        # In PPO we do multiple backward passes on a single rollout's state-action pairs
        # To ensure stable training, we limit how much the probability of a given action is allowed to change over the k iterations
        # https://huggingface.co/learn/deep-rl-course/unit8/visualize

        for k in range(ppo_num_training_steps_per_epoch):
            # re-sample probability of the taken actions with newly updated model
            probs, new_values = model.evaluate_new_prob(observations, actions)
            new_values = new_values.squeeze(-1)

            # clipped surrogate objective (vectorised over all steps and envs)
            prob_ratio = torch.exp(probs - old_probs)
            clipped_ratio = torch.clamp(prob_ratio, 1 - ppo_clip_size, 1 + ppo_clip_size)
            total_loss = -torch.min(prob_ratio * advantages, clipped_ratio * advantages).mean()

            # value loss - the delta between the estimated value and the actual future returns
            total_value_loss = ((new_values - returns) ** 2).mean()

            # combine to total loss (subtract entropy to reward exploration)
            final_loss = total_loss + value_loss_coeff * total_value_loss

            # backpropagate and update weights
            model.backward(final_loss)

        # log mean total rewards per rollout to tensorboard
        print(f"rollout {n*num_envs} / {n_rollouts*num_envs} : actor loss = {total_loss.item()} : critic loss = {total_value_loss.item()} : reward = {torch.stack(rewards).sum(dim=0).mean().item()}")
        writer.add_scalar("Reward / Total reward (mean)", torch.stack(rewards).sum(dim=0).mean().item(), n*num_envs)
        writer.add_scalar("loss/policy", total_loss.item(), n*num_envs)
        writer.add_scalar("loss/value", total_value_loss.item(), n*num_envs)

        # save the model weights locally
        torch.save(model.state_dict(), "ppo_ant.pth")

# set up tensorboard log session
run_name = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f") + "_MY_PPO"
writer = SummaryWriter(log_dir=f"runs/torch/{task_name}/{run_name}")

# train model
train()
