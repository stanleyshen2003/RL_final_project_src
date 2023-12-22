from td3_agent_CarRacing import CarRacingTD3Agent
import argparse
import json
import numpy as np
import requests


def connect(agent, url: str = 'http://localhost:5000'):
    while True:
        # Get the observation
        response = requests.get(f'{url}')
        if json.loads(response.text).get('error'):
            print(json.loads(response.text)['error'])
            break
        obs = json.loads(response.text)['observation']
        obs = np.array(obs).astype(np.uint8)

        # Decide an action based on the observation (Replace this with your RL agent logic)
        
        action_to_take = agent.act(obs)  # Replace with actual action
        #action_to_take[0] /=2
        # Send an action and receive new observation, reward, and done status
        response = requests.post(f'{url}', json={'action': action_to_take.tolist()})
        if json.loads(response.text).get('error'):
            print(json.loads(response.text)['error'])
            break

        result = json.loads(response.text)
        terminal = result['terminal']

        if terminal:
            print('Episode finished.')
            return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', type=str, default='http://localhost:5000', help='The url of the server.')
    args = parser.parse_args()
    config = {
        "gpu": True,
        "training_steps": 1e8,
        "gamma": 0.96, #0.99
        "tau": 0.005,
        "batch_size": 32,
        "warmup_steps": 1000,
        "total_episode": 10000,
        "lra": 1e-4,
        "lrc": 1e-4,
        "replay_buffer_capacity": 5000,
        "logdir": 'log/CarRacing',
        "update_freq": 3,
        "eval_interval": 5,
        "eval_episode": 1,
    }
    agent = CarRacingTD3Agent(config)
    agent.load('log/CarRacing/td3_mod/model_498501_214.pth')
    connect(agent, url=args.url)
