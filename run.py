import gym
import time
import random
import numpy as np
import chula_rl as rl
from agent import Agent
import matplotlib.pyplot as plt


def create_env():
    env = gym.make('CartPole-v1')
    env.reset()
    return env


def run_one_step_batch(agent, env, n_max_interaction=np.inf, max_iter=3000, end_score=None,
                       train=True, load_checkpoint=False, verbose=False):

    if n_max_interaction == np.inf and end_score == None:
        raise ValueError(
            "Must at least gives n_max_interaction or end_score to the function")

    best_score = env.envs[0].reward_range[0]
    avg_score = env.envs[0].reward_range[0]
    score_history = []
    avg_score_history = []

    if load_checkpoint:
        agent.load_models()

    explorer = rl.explorer.VecOneStepExplorer(
        n_max_interaction=n_max_interaction, env=env)
    score = [0] * agent.batch_size
    episode = 0
    iteration = 0

    print(
        f"Start training one step discrete for {n_max_interaction} steps ...")
    while True:
        try:
            data = explorer.step(agent)
            rewards, dones = data['r'], data['done']
            score += rewards

            if train:
                agent.optimize_step(data)

            # If any parallel env finished an ep
            if any(dones):
                finished_idxs = []
                for i in range(len(dones)):
                    if dones[i]:
                        # keep the finished index so we can reset score later
                        finished_idxs.append(i)

                        # append the finished score
                        score_history.append(score[i])

                avg_score = np.mean(score_history[-100:])

                if avg_score > best_score:
                    best_score = avg_score

                if not load_checkpoint:
                    # agent.save_models()
                    pass

                avg_score_history.append(avg_score)

                if verbose:
                    print(
                        f"Episode {episode+1} score_last_10: {np.mean(score_history[-10:])}, avg_score: {avg_score:.3f}, best_score: {best_score:.3f}")

                episode += len(finished_idxs)

                # reset score that are finished
                for idx in finished_idxs:
                    score[idx] = 0

            if avg_score > end_score:
                print(
                    f"Achieved {avg_score} average score (> {end_score} end_score)")
                break

            if (iteration + 1) % 100 == 0:
                print(
                    f"Iteration {iteration + 1} Episode {episode+1} score_last_10: {np.mean(score_history[-10:])}, avg_score: {avg_score:.3f}, best_score: {best_score:.3f}")

            if iteration > max_iter:
                print(f"max iteration {iteration} reached")
                break

            iteration += 1

        except rl.exception.InteractionExceeded:
            print(f'Completed {n_max_interaction} steps')
            break

    res = {
        'iteration': iteration,
        'episode_elapsed': episode,
        'best_score': best_score,
        'score_history': score_history,
        'avg_score_history': avg_score_history,
    }

    return res


def run(batch_size, lr=1e-5, n_max_interaction=np.inf, max_iter=3000, end_score=50, discount_factor=0.99, fc1_dims=1024, fc2_dims=512):
    env = rl.env.DummyVecEnv([create_env] * batch_size)
    env.reset()
    features_dim = env.envs[0].reset().shape[0]
    agent = Agent(
        features_dim=features_dim,
        batch_size=batch_size,
        lr=lr,
        discount_factor=discount_factor,
        n_actions=env.action_space.n,
        fc1_dims=fc1_dims,
        fc2_dims=fc2_dims,
    )

    start_time = time.time()

    res = run_one_step_batch(
        agent,
        env,
        n_max_interaction=n_max_interaction,
        max_iter=max_iter,
        end_score=end_score,
        train=True,
        load_checkpoint=False,
        verbose=False
    )

    time_taken = time.time() - start_time
    print(f"time_taken: {time_taken:.3f} seconds")

    return res


def main():
    pass


def run_param_search(learning_rates=[5e-3, 1e-3], batch_size=16, end_score=200):
    plt.figure(figsize=(15, 5))

    def random_color():
        return tuple([random.random() for _ in range(3)])

    colors = [random_color() for _ in range(len(learning_rates))]

    for i, lr in enumerate(learning_rates):
        for j in range(1):
            print(f"Running lr={lr} attempt {j+1}")
            res = run(batch_size=batch_size, lr=lr, end_score=end_score)
            plt.plot(res['avg_score_history'],
                     label="lr=" + str(lr), color=colors[i])

    plt.legend()
    plt.show()
    plt.savefig('plot.png')


if __name__ == "__main__":
    run_param_search()
