import matplotlib.style as style
import numpy as np

from matplotlib import pyplot as plt

style.use('seaborn')

with open("elmo_original_sst5.txt", 'r') as f:
    fig, axs = plt.subplots(5, sharex=True, sharey=False)
    idx = 0
    for line in f:
        experiment = line.split(",")
        data = experiment[::-1]
        num_points = [x for x in range(len(data))]
        performance = np.asarray([float(x) for x in data])
        # performance = (performance-min(performance))/(max(performance)-min(performance))

        x_labels = np.arange(len(num_points))
        axs[idx].bar(x_labels, performance, align='center', alpha=0.9, width=0.9)

        ylabel = 'Level'
        if idx == 4:
            ylabel = 'Leaf'
        elif idx == 3:
            ylabel = 'Parent'
        elif idx == 2:
            ylabel = 'GParent'
        elif idx == 1:
            ylabel = 'GGParent'
        elif idx == 0:
            ylabel = 'Root'

        axs[idx].grid(True, axis='x')
        axs[idx].grid(False, axis='y')
        axs[idx].set_xticks(x_labels)
        axs[idx].set(xlabel='Layer', ylabel=ylabel)
        axs[idx].label_outer()
        axs[idx].set_yticklabels([])


        idx += 1

    fig.suptitle('Normalized ELMo (original) Ancestor SST-5 Performance Per-Layer')

    plt.show()

