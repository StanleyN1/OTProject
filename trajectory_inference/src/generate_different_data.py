from stanley_appex.estimation import *
from stanley_appex.plotting import *
from stanley_appex.utils import *
from stanley_appex.generate_data import *
import numpy as np
import matplotlib.pyplot as plt
import os

np.random.seed(0)

d = 2
A_pos = np.array([[1.0, 0.0], 
                  [0.0, 1.0]])

A_neg = np.array([[-1.0, 0.0], 
                  [0.0, -1.0]])

A_rot = np.array([[0.0, -1.0],
                  [1.0, 0.0]])

A_zero = np.array([[-1.0, 0.0],
                   [-1.0, 0.0]])

G = np.array([[1.0, 0.0], 
              [0.0, 1.0]])

G_mixed = np.array([[1.0, 1.0], 
                    [1.0, 1.0]])

As = {'pos':A_pos, 'neg':A_neg, 'rot':A_rot, 'zero':A_zero}
Gs = {'iden': G, 'mixed': G_mixed}

growth_rate = 0.0
N_init = 150 # 5
X0 = np.random.multivariate_normal(np.zeros(d), np.diag(np.ones(d)), N_init) # np.random.normal(0, 1, (N_init, A.shape[1]))
# X0 = np.random.normal(0, 1, (N_init))
Nt = 15000
dt = 0.0001
print("Final Time", Nt*dt)


output_dir = 'data/3_30_2025'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for A_name, A in As.items():
    for G_name, G in Gs.items():
        H = G@G.T
        process = BranchingStochasticProcess(A, G, dt=dt, Nt=Nt, N_traj=N_init)
        process.simulate(X0, growth_rate=growth_rate)
        print("Final number of trajectories:", process.N_traj)
        process.save_file(f'data/3_30_2025/trajectories_A={A_name}_G={G_name}.h5')
        print('Saved file', f'data/3_30_2025/trajectories_A={A_name}_G={G_name}.h5')