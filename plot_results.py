import matplotlib.style as style
import numpy as np

from matplotlib import pyplot as plt

style.use('seaborn')

with open("lm_probing_results.txt", "r") as f:
    fig, axs = plt.subplots(5, sharex=True, sharey=True)
    idx = 0
    for line in f:
        experiment = line.split(",")
        data = experiment[1:]
        num_points = [x for x in range(len(data))]
        performance = np.asarray([float(x) for x in reversed(data)])
        performance = (performance-min(performance))/(max(performance)-min(performance))

        x_labels = np.arange(len(num_points))
        axs[idx].bar(x_labels, performance, align='center', alpha=0.8, width=0.9)

        axs[idx].grid(True, axis='x')
        axs[idx].grid(False, axis='y')
        axs[idx].set_xticks(x_labels)
        axs[idx].set(xlabel='Layer', ylabel='Level')
        axs[idx].label_outer()

        idx += 1

    plt.show()
