import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

desired_type = 'SED'  # SED or DOA
save_graph = True
show_graph = True

dir_path = '/home/username/results_logs/'

if desired_type == 'SED':
    # SED loss plots #################################
    loss_path = os.path.join(dir_path, 'losses_SED.pickle')
    with open(loss_path, 'rb') as handle:
        losses = pickle.load(handle)

    idx = 1
    for curr_loss in losses:
        fig = plt.figure(num=idx, figsize=(5, 5))
        train = curr_loss['Train SED loss']
        valid = curr_loss['Valid SED loss']

        batches = 200 * np.arange(0, len(train))
        plt.plot(batches, train, batches, valid, 'r')
        plt.xlabel('Batches')
        plt.ylabel('Loss')
        plt.title('SED losses for trial ' + str(idx))
        plt.legend(['Train loss', 'Validation loss'])
        plt.axis([-1, batches[-1] + 100, 0, 0.4])
        plt.grid(True)
        if save_graph:
            fig_name = 'sed_figure_' + str(idx) + '.jpg'
            fig_path = os.path.join(dir_path, fig_name)
            plt.savefig(fig_path)
        idx += 1

    if show_graph:
        plt.show()

elif desired_type == 'DOA':
    # DOA loss plots #################################
    loss_path = os.path.join(dir_path, 'losses_DOA.pickle')
    with open(loss_path, 'rb') as handle:
        losses = pickle.load(handle)

    idx = 1
    for curr_loss in losses:
        fig = plt.figure(num=idx, figsize=(5, 5))
        train = curr_loss['Train DOA loss']
        valid = curr_loss['Valid DOA loss']
        batches = 200 * np.arange(0, len(train))
        plt.plot(batches, train, batches, valid, 'r')
        plt.xlabel('Batches')
        plt.ylabel('Loss')
        plt.title('DOA losses for trial ' + str(idx))
        plt.legend(['Train loss', 'Validation loss'])
        plt.axis([-1, batches[-1] + 100, 0, 3])
        plt.grid(True)
        if save_graph:
            fig_name = 'doa_figure_' + str(idx) + '.jpg'
            fig_path = os.path.join(dir_path, fig_name)
            plt.savefig(fig_path)
        idx += 1

    if show_graph:
        plt.show()

else:
    print('Error: unknown loss type.')


