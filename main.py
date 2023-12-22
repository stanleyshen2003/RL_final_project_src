from td3_agent_CarRacing import CarRacingTD3Agent

if __name__ == '__main__':
	# my hyperparameters, you can change it as you like
	config = {
		"gpu": True,
		"training_steps": 1e8,
		"gamma": 0.9, #0.99
		"tau": 0.005,
		"batch_size": 64,
		"warmup_steps": 1000,
		"total_episode": 5000,
		"lra": 4.5e-5,
		"lrc": 4.5e-5,
		"replay_buffer_capacity": 5000,
		"logdir": 'log/CarRacing/td3_train',
		"update_freq": 3,
		"eval_interval": 5,
		"eval_episode": 1,
	}
	agent = CarRacingTD3Agent(config)
	#agent.load('log/CarRacing/td3_mod/model_1089204_48.pth')  #1
	#agent.load('log/CarRacing/td3_test/model_72656_1.pth')  #2345
	agent.train()


