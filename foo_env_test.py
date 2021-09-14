import gym
#import gym_foo
#env = gym.make('foo-v0')
env = gym.make('Pendulum-v0')


state = env.reset()
next_state, reward, done, info = env.step(env.action_space.sample())
print('state: {}, reward: {}, done: {}, info: {}, next_state: {}'.format(state, reward, done, info, next_state))
print('action_space:{}'.format(env.action_space))	 
print('observation_space:{}'.format(env.observation_space))	 


max_ep = 10
for ep_cnt in range(max_ep):
	step_cnt = 0
	ep_reward = 0
	done = False
	state = env.reset()
	
	while not done:
		next_state, reward, done, _ = env.step(env.action_space.sample())
		env.render()
		step_cnt += 1
		ep_reward += reward
		state = next_state
	
	print('Episode: {}, step count: {}, episode reward: {}'. format(ep_cnt, step_cnt, ep_reward))
env.close()

