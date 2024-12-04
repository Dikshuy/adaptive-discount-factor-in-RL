import pickle
import numpy as np
import matplotlib.pyplot as plt

# gammas and q inits
gammas = [0.99, 0.95, 0.8, 0.5, 0.1]
q_initializations = [100.0, 20.0, 10.0, 1.0, -5.0]

base_path = 'experiments/old_actor_critic_results/mountain_car/mountain-car_init_'

for gamma in gammas:
    all_mean_eval_returns = {}
    all_std_eval_returns = {}
    all_mean_eval_lengths = {}
    all_std_eval_lengths = {}
    
    for q_init in q_initializations:
        file_path = f'{base_path}{q_init}_gamma_{gamma}_results.pkl'
        
        try:
            with open(file_path, 'rb') as file:
                results = pickle.load(file)

            eval_returns = np.array(results['evaluation returns'])
            eval_lengths = np.array(results['evaluation lengths'])

            # Mean and standard deviation across seeds
            all_mean_eval_returns[q_init] = eval_returns.mean(axis=0)
            all_std_eval_returns[q_init] = eval_returns.std(axis=0)

            all_mean_eval_lengths[q_init] = eval_lengths.mean(axis=0)
            all_std_eval_lengths[q_init] = eval_lengths.std(axis=0)
        
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            continue

    # evaluation returns
    plt.figure()
    for q_init, mean_eval_returns in all_mean_eval_returns.items():
        std_eval_returns = all_std_eval_returns[q_init]
        plt.plot(mean_eval_returns, label=f'Q_init = {q_init}')
        plt.fill_between(range(len(mean_eval_returns)), 
                         mean_eval_returns - std_eval_returns,
                         mean_eval_returns + std_eval_returns, alpha=0.3)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title(f'Evaluation Returns (gamma = {gamma}, mountain-car)')
    plt.legend()
    plt.show()

    # evaluation lengths
    plt.figure()
    for q_init, mean_eval_lengths in all_mean_eval_lengths.items():
        std_eval_lengths = all_std_eval_lengths[q_init]
        plt.plot(mean_eval_lengths, label=f'Q_init = {q_init}')
        plt.fill_between(range(len(mean_eval_lengths)), 
                         mean_eval_lengths - std_eval_lengths,
                         mean_eval_lengths + std_eval_lengths, alpha=0.3)
    plt.xlabel('Episodes')
    plt.ylabel('Lengths')
    plt.title(f'Evaluation Lengths (gamma = {gamma}, mountain-car)')
    plt.legend()
    plt.show()