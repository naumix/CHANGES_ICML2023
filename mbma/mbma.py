import os
import random
import time
from ppo_utils import Agent, evaluate, evaluate2, RewardNormalizer, ExperienceBuffer, transition_net, reward_net, generate_trajectory, generate_trajectory2, calculate_advantage, view_chunk, ObsNormalizer, get_returns

import dmc2gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from args import parse_args

os.environ['MUJOCO_GL'] = 'disable'

def main():
    args = parse_args()
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb
        wandb.init(project=args.wandb_project_name,entity=args.wandb_entity,sync_tensorboard=True,config=vars(args), name=run_name,monitor_gym=True,save_code=True,)
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text("hyperparameters","|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),)
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    for experiment in range(args.experiment_repeats):
        envs = dmc2gym.make(domain_name=args.gym_id,task_name=args.task_name, seed=args.seed, visualize_reward=False, from_pixels=False, frame_skip=args.action_repeat)
        agent = Agent(envs, args).to(device)
        t_net = transition_net(envs, 2*args.hidden_dim).to(device)
        r_net = reward_net(envs, 2*args.hidden_dim).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
        t_optimizer = optim.Adam(t_net.parameters(), lr=args.wm_learning_rate, eps=1e-8)
        r_optimizer = optim.Adam(r_net.parameters(), lr=args.wm_learning_rate, eps=1e-8)
        if args.normalize_rewards:
            normalizer = RewardNormalizer(device)
        buffer = ExperienceBuffer(args.buffer_size, envs, device)
        if args.normalize_states:
            s_normalizer = ObsNormalizer(envs.observation_space.shape, device)
        wm_activated = False
        wm_step = 0
        first = True
        # ALGO Logic: Storage setup
        obs = torch.zeros((args.num_steps, args.num_envs) + envs.observation_space.shape).to(device)
        actions = torch.zeros((args.num_steps, args.num_envs) + envs.action_space.shape).to(device)
        logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
        rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
        dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
        values = torch.zeros((args.num_steps, args.num_envs)).to(device)
        # TRY NOT TO MODIFY: start the game
        global_step = 0
        start_time = time.time()
        next_obs = envs.reset()
        if args.normalize_states:
            s_normalizer.add(next_obs)
        next_obs = torch.Tensor(next_obs).float().to(device)
        next_done = torch.zeros(args.num_envs).to(device)
        num_updates = args.total_timesteps // args.batch_size
        episode_reward = 0
        for update in range(1, num_updates + 1):
            # Annealing the rate if instructed to do so.
            if args.anneal_lr:
                frac = 1.0 - (update - 1.0) / (args.anneal_ratio*num_updates)
                lrnow = frac * args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow
            for step in range(0, args.num_steps):
                global_step += 1 * args.num_envs
                obs[step] = next_obs
                dones[step] = next_done
                # ALGO LOGIC: action logic
                with torch.no_grad():
                    input_ = next_obs.unsqueeze(0).float()
                    if args.normalize_states:
                        input_ = s_normalizer.forward(input_)
                    action, logprob, _, value = agent.get_action_and_value(input_)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob
                # TRY NOT TO MODIFY: execute the game and log data.
                state_ = next_obs.clone()
                next_obs, reward, done, info = envs.step(action.cpu().clip(min=-1, max=1).squeeze(0).numpy())
                buffer.add(state_, action.clip(min=-1.0, max=1.0), reward, torch.tensor(next_obs))
                if args.normalize_rewards:
                    normalizer.add(reward)
                episode_reward += reward
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                if args.normalize_states:
                    s_normalizer.add(next_obs)
                next_obs = torch.Tensor(next_obs).float().to(device)
                next_done = torch.Tensor(np.array(done)).to(device)
                if done:
                    next_obs = envs.reset()
                    if args.normalize_states:
                        s_normalizer.add(next_obs)
                    next_obs = torch.Tensor(next_obs).float().to(device)
                    episode_reward = 0
            # bootstrap value if not done
            if args.normalize_rewards:
                normalizer.get_meanstd()
                rewards = normalizer.forward(rewards)
            with torch.no_grad():
                input_ = next_obs.unsqueeze(0)
                if args.normalize_states:
                    input_ = s_normalizer.forward(input_)
                next_value = agent.get_value(input_).reshape(1, -1)
            returns, advantages = get_returns(args, rewards, values, dones, next_done, next_value, device)
            # flatten the batch
            b_obs = obs.reshape((-1,) + envs.observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)
            if global_step > args.wm_init_steps:
                wm_activated = True
                wm_step += 1
                num_actions = int(wm_step * args.action_samples // (num_updates * args.anneal_limit) + 1) 
                num_horizon = int(wm_step * args.horizon // (num_updates * args.anneal_limit) + 2)
                if num_actions > args.action_samples:
                    num_actions = args.action_samples
                if num_horizon > args.horizon:
                    num_horizon = args.horizon
            # Optimizing the policy and value network
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            if wm_activated:
                wm_epochs = int(args.update_epochs * args.minibatch_size)
                if first:
                    wm_epochs = int(args.update_epochs * args.minibatch_size * 8)
                    first = False
            if global_step > args.wm_init_steps:
                for wm_batch in range(wm_epochs):
                    s_, a_, r_, ns_ = buffer.sample(args.wm_batch_size)
                    ns_pred1 = t_net(s_, a_)
                    transition_loss = F.mse_loss(ns_pred1, ns_)
                    t_optimizer.zero_grad()
                    transition_loss.backward()
                    t_optimizer.step()
                    r_pred1 = r_net(s_, a_)
                    reward_loss = F.mse_loss(r_pred1, r_)
                    r_optimizer.zero_grad()
                    reward_loss.backward()
                    r_optimizer.step()
            if wm_activated:
                adv = torch.zeros((b_obs.size(0), num_actions)).float().to(device)
                imagined_lps = torch.zeros((b_obs.size(0), num_actions)).float().to(device)
                imagined_as = torch.zeros((b_obs.size(0), b_actions.size(1), num_actions)).float().to(device)
                for sample in range(num_actions):
                    if args.normalize_states:
                        s_, a_, r_, fsv_, lp_, v_ = generate_trajectory(b_obs, num_horizon, t_net, r_net, agent, device, s_normalizer)
                    else:
                        s_, a_, r_, fsv_, lp_, v_ = generate_trajectory2(b_obs, num_horizon, t_net, r_net, agent, device)
                    advantages_, returns_ = calculate_advantage(args, fsv_, r_, v_, device, normalizer)
                    adv[:, sample] = advantages_[:, 0]
                    imagined_lps[:, sample] = lp_[:, 0]
                    s_ = s_[:,:,0]
                    imagined_as[:,:,sample] = a_[:,:,0]
                b_logprobs = torch.cat([b_logprobs.unsqueeze(1), imagined_lps], 1)
                b_advantages = torch.cat([b_advantages.unsqueeze(1), adv], 1)
                if args.normalize_states:
                    s_ = s_normalizer.forward(s_)
            if args.normalize_states:
                b_obs = s_normalizer.forward(b_obs)
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]
                    if wm_activated:
                        batch_obs_ = b_obs[mb_inds]
                        batch_as_ = b_actions[mb_inds]
                        i_ = 0
                        for i_ in range(num_actions):
                            batch_obs_ = torch.cat([batch_obs_, s_[mb_inds]], 0)
                            batch_as_ = torch.cat([batch_as_, imagined_as[mb_inds, :, i_]], 0)
                        _, newlogprob, entropy, newvalue, = agent.get_action_and_value(batch_obs_, batch_as_)
                        newvalue = view_chunk(newvalue, num_actions+1, 0).squeeze()[:,0]
                        newlogprob = view_chunk(newlogprob.unsqueeze(1), num_actions+1, 0).squeeze()
                        logratio = newlogprob - b_logprobs[mb_inds, :]
                        ratio = logratio.exp()
                        mb_advantages = b_advantages[mb_inds, :]
                    else:
                        _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                        logratio = newlogprob - b_logprobs[mb_inds]
                        ratio = logratio.exp()
                        mb_advantages = b_advantages[mb_inds]
                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
                    #mb_advantages = b_advantages[mb_inds, :] if args.width_sampling and wm_activated else b_advantages[mb_inds] 
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    if wm_activated:
                        pg_loss = torch.max(pg_loss1, pg_loss2).mean(1).mean()
                    else:
                        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef)
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()
                if args.target_kl is not None:
                    if approx_kl > args.target_kl:
                        break
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            if update % 5 == 0:
                if args.normalize_states:
                    eval_rew = evaluate2(args, agent, 4, device, s_normalizer)
                else:
                    eval_rew = evaluate(args, agent, 4, device)
                print(f"global_step={global_step}, eval_return={eval_rew}, SPS={int(global_step / (time.time() - start_time))}")
            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
    
        envs.close()
        writer.close()
        if experiment == 0:
            global_results = np.zeros((len(agent.record), args.experiment_repeats))
        global_results[:, experiment] = np.array(agent.record)
        name_ = 'results_ppo_w_' + str(args.gym_id) + '_' + str(args.task_name)
        np.save(name_, global_results)
    
main()
