import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

save_graph = True
show_graph = True

dir_path = '/home/username/results_logs/'

sed_file = os.path.join(dir_path, 'losses_SED.pickle')
doa_file = os.path.join(dir_path, 'losses_DOA.pickle')

with open(sed_file, 'rb') as handle:
    sed_loss = pickle.load(handle)

with open(doa_file, 'rb') as handle:
    doa_loss = pickle.load(handle)

idx = 1

sed_graph = np.array(sed_loss[0]['Valid SED loss'])
doa_graph = np.array(doa_loss[0]['Valid DOA loss'])

fig = plt.figure(num=1, figsize=(5, 5))
batches = 200 * np.arange(0, len(sed_graph))

plt.plot(batches, sed_graph, batches, doa_graph, 'r')
plt.xlabel('Batches')
plt.ylabel('Loss')
plt.title('Validation losses for lstm')
plt.legend(['SED validation loss', 'DOA validation loss'])
plt.axis([-1, batches[-1] + 100, 0, 2])
plt.grid(True)

if save_graph:
    fig_name = 'validation_losses.jpg'
    fig_path = os.path.join(dir_path, fig_name)
    plt.savefig(fig_path)

if show_graph:
    plt.show()



