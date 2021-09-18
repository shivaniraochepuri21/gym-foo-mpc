import gym
import gym_foo
import matplotlib.pyplot as plt

env = gym.make('foo-v0')

step_cnt = 0
ep_reward = 0
done = False
state = env.reset()
env.render()
ep_cnt = 0
y_ref = 0.00
y_mpc = []
rewards_mpc = []

#while not done:
for i in range(200):
	mpc = env.get_mpc()
	x0 = env.get_x0()
	
	#u0 shape is (1,1) i.e [[.]]
	u0 = mpc.make_step(x0)
	action = u0
	
	next_state, reward, done, _ = env.step(action)
	y = env.get_y()
	y_mpc.append(y)
	
	env.render()
	step_cnt += 1
	ep_reward += reward
	rewards_mpc.append(reward)
	state = next_state
	
	if abs(y-y_ref) <= 0.001:
		done = True
		    
	env.update_mpc(mpc)	

print('total number of time steps: ', i)

#print('Episode: {}, step count: {}, episode reward: {}'. format(ep_cnt, step_cnt, ep_reward))
env.close()

print(y_mpc)
print(len(y_mpc))
print(rewards_mpc)
print(len(rewards_mpc))
plt.figure(1)
plt.plot(rewards_mpc)
plt.title("rewards mpc")
plt.show()

plt.figure(2)
plt.plot(y_mpc)
plt.plot(y_ref)
plt.title("theta tracking by mpc")
plt.show()
