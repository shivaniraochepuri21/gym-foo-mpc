import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from os import path
import sys
from casadi import *
import do_mpc

#continuous dynamic model for a simple inverted pendulum
model_type = 'continuous'
model = do_mpc.model.Model(model_type)

#g = 9.80665
g = 10.0
m = 1.0
l = 1.0

#model variable - theta, dtheta, u or force or torque
theta = model.set_variable('_x',  'theta')
dtheta = model.set_variable('_x',  'dtheta')
u = model.set_variable('_u',  'force')
model.set_rhs('theta', dtheta)
model.set_rhs('dtheta', -3 * g / (2 * l) * np.sin(-np.pi + theta) + 3.0 / (m * l ** 2) * u )

E_pot = (1/2) * m * g * l * np.cos(theta)
E_kin = (1/6) * m * l**2 * dtheta**2 
model.set_expression('E_kin', E_kin)
model.set_expression('E_pot', E_pot)
model.setup()

#controller
mpc = do_mpc.controller.MPC(model)
setup_mpc = {
    'n_horizon': 10,
    'n_robust': 1,
    'open_loop': 0,
    't_step': 0.05, #previously 1.0
    'state_discretization': 'collocation',
    'collocation_type': 'radau',
    'collocation_deg': 2,
    'collocation_ni': 2,
    'store_full_solution': True,
    # Use MA27 linear solver in ipopt for faster calculations:
    #'nlpsol_opts': {'ipopt.linear_solver': 'MA27'}
}
mpc.set_param(**setup_mpc)

mterm = model.aux['E_kin'] - model.aux['E_pot'] # terminal cost
lterm = model.aux['E_kin'] - model.aux['E_pot'] # stage cost
mpc.set_objective(mterm=mterm, lterm=lterm)
mpc.set_rterm(force=0.001)

#constriants
#max_speed = 8 max_torque = 2.0 dt = 0.05
mpc.bounds['lower', '_x', 'theta'] = np.radians(0) #radians
mpc.bounds['lower', '_x', 'dtheta'] = -8 #radians/sec 
mpc.bounds['upper', '_x','theta'] = np.radians(360) #radians
mpc.bounds['upper', '_x','dtheta'] = 8 #radians/sec
mpc.bounds['lower','_u','force'] = -2.0 #newtons meter
mpc.bounds['upper','_u','force'] = 2.0 #newtons meter
mpc.setup()

#simulator
estimator = do_mpc.estimator.StateFeedback(model)
simulator = do_mpc.simulator.Simulator(model)
params_simulator = {
    'integration_tool': 'cvodes',
    'abstol': 1e-10,
    'reltol': 1e-10,
    't_step': 0.05
}
simulator.set_param(**params_simulator)
simulator.setup()

#closed loop simulation
simulator.x0['theta'] = 0.99*np.pi
x0 = simulator.x0.cat.full()

mpc.x0 = x0
estimator.x0 = x0
mpc.set_initial_guess()

mpc.reset_history()

class FooEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, g=10.0):
        self.max_speed = 8
        self.max_torque = 2.0
        self.dt = 0.05
        self.g = g
        self.m = 1.0
        self.l = 1.0
        self.viewer = None
        self.x0 = x0
        self.y_next = x0
        self.y = 0.99*np.pi
        
        high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_mpc(self):
        return mpc
	
    def update_mpc(self, mpc_new):
        mpc = mpc_new
        
    def get_x0(self):
        return self.x0
    
    def isDone(self):
    	flag = False
    	if np.linalg.norm(np.array(self.x0[0])-np.array(self.y_next[0])) <= 0.01:
    		flag = True
    	print('flag',flag)
    	return flag
		
    def	get_y(self):
        return self.y
			
    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        
        PE = (1/2) * m * g * l * np.cos(th)
        KE = (1/6) * m * l**2 * thdot**2 
        costs = KE - PE + 0.001 * (u**2)

        #u0 = mpc.make_step(x0)
        #print('u_in_step',u)
        #print('type',type(u))
        
        u = u.reshape((1,1))
        y_next = simulator.make_step(u)
        x0 = estimator.make_step(y_next)
        self.state = np.ravel(np.array([x0[0],x0[1]]))
        self.x0 = x0
        self.y_next = y_next
        self.y = y_next[0][0]
        print('x0, y_next', x0, y_next)
        return self._get_obs(), -costs, False, {}

    def reset(self):
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])
	
    def render(self, mode="human"):
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, 0.2)
            rod.set_color(0.8, 0.3, 0.3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(0.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1.0, 1.0)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi
    
