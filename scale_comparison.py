from lqr2_scale import LQRCompare
import numpy as np
import matplotlib.pyplot as plt
import time

s_scales = [2, 4, 8, 16, 32, 64, 128, 256]
s_scales = [64, 32, 16, 8, 4]
# s_scales = np.array(s_scales)[::-1]

method_names = [
    'classical', 
    'quantum',
]

results = {m: [] for m in method_names} 

for method in method_names:
	for s in s_scales:
		s_time = time.time()

		sys = LQRCompare(s=s, q=9, T=10, L=10, seed=3, n_epoch=1, N_Sample_J=1, name=method)
		sys.run()


		e_time = time.time()

		elapsed = e_time - s_time
		print(method, s, elapsed)
		results[method].append(elapsed)

print("start ploting")

plt.tight_layout()


### save
plt.figure(figsize=(8, 6))
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20


plt.clf()
for i_name in range(len(method_names)):
    method_name = method_names[i_name]

    x = np.array(s_scales)
    color = f'C{i_name}'

    display_name = method_name
    if method_name == 'quantum':
        display_name = 'ours'

    plt.plot(x, results[method_name], color=color, linewidth=4, label=display_name)


plt.yscale('log', base=10)
plt.xlabel('DoF', fontsize=30, labelpad=-9)
plt.ylabel('Time (s)', fontsize=30, labelpad=-12)
plt.legend(fontsize=30)
plt.grid()
plt.savefig(f'scale{s_scales[0]}.png')



