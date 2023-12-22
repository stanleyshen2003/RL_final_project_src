# from td3_agent_CarRacing import CarRacingTD3Agent

# if __name__ == '__main__':
# 	# my hyperparameters, you can change it as you like
# 	config = {
# 		"gpu": True,
# 		"training_steps": 1e8,
# 		"gamma": 0.9, #0.99
# 		"tau": 0.005,
# 		"batch_size": 64,
# 		"warmup_steps": 1000,
# 		"total_episode": 5000,
# 		"lra": 4.5e-5,
# 		"lrc": 4.5e-5,
# 		"replay_buffer_capacity": 5000,
# 		"logdir": 'log/CarRacing/td3_train',
# 		"update_freq": 3,
# 		"eval_interval": 5,
# 		"eval_episode": 1,
# 	}
# 	agent = CarRacingTD3Agent(config)
# 	#agent.load('log/CarRacing/td3_mod/model_1089204_48.pth')  #1
# 	#agent.load('log/CarRacing/td3_test/model_72656_1.pth')  #2345
# 	agent.train()


from hyperopt import fmin, tpe, hp, Trials
from td3_agent_CarRacing import CarRacingTD3Agent
import os
# Define the hyperparameter search space
space = {
    "gamma": hp.uniform("gamma", 0.9, 0.999),
    "tau": hp.uniform("tau", 0.001, 0.1),
    "lra": hp.loguniform("lra", -7, -3),
    "lrc": hp.loguniform("lrc", -7, -3),
    "update_freq": hp.choice("update_freq", [2, 3, 4]),
}

def objective(params):
    # Function to optimize (train and evaluate the agent with given hyperparameters)
    agent_config = {
        "gpu": True,
        "training_steps": 1e8,
        "warmup_steps": 1000,
        "total_episode": 2000,
        "batch_size": 64,
        "replay_buffer_capacity": 5000,
        "eval_interval": 5,
        "eval_episode": 10,
        "logdir": 'log/CarRacing/td3_train'
    }
    agent_config.update(params)
    
    agent = CarRacingTD3Agent(agent_config)
    # Train the agent
    # You may want to track some performance metric to minimize or maximize here
    # For example, the average reward over several episodes
    agent.train()
    agent.save(os.path.join(agent.writer.log_dir, f"model_{agent.total_time_step}_{agent.lrc}_{agent.lra}.pth"))

    # Dummy return (replace with the actual metric you want to optimize)
    return agent.evaluate()  # Modify this to return the metric you want to optimize

# Initialize the trials object to track the results
trials = Trials()

# Run hyperparameter optimization using Bayesian optimization
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=15,  # Set the number of evaluations/trials
    trials=trials
)

print("Best hyperparameters:", best)


