import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from options import CompareModelsOptions
import glob
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from scipy.stats import sem

GRAPHS_FOLDER = 'graphs'
STATS_FOLDER = 'stats'
STAT_EXTENSION = '.npy'


def main():
    args = CompareModelsOptions().parse()
    experiments = args.experiments.split(',')
    exp_labels = args.exp_labels.split(',') if args.exp_labels is not None else None
    stats_names = set()
    colors = plt.cm.rainbow(np.linspace(0, 1, len(experiments)))
    graphs_folder = os.path.join(GRAPHS_FOLDER, args.experiments)
    if not os.path.exists(graphs_folder):
        os.mkdir(graphs_folder)
    exps_mean_classification_acc = []
    exps_mean_clustering_acc = []
    exps_classification_ste = []
    exps_clustering_ste = []
    for i, exp in enumerate(experiments):
        exp_folder = glob.glob('{}*'.format(os.path.join(args.base_folder, exp)))[0]
        sub_exps = sorted(glob.glob('{}/[0-9]*'.format(exp_folder)))
        stat_folders = [os.path.join(sub_exp, STATS_FOLDER) for sub_exp in sub_exps] if len(sub_exps) > 0 \
            else [os.path.join(exp_folder, STATS_FOLDER)]

        exp_classification_acc = []
        exp_clustering_acc = []
        stats_dict = defaultdict(list)
        for j, stat_folder in enumerate(stat_folders, 1):
            stat_index = -1
            if args.lowest_loss:
                stat_index = np.argmin(np.load(os.path.join(stat_folder, 's_labeled_loss.npy')))
            for stat_file in os.listdir(stat_folder):
                if stat_file.endswith(STAT_EXTENSION):
                    stat_name = stat_file.split(STAT_EXTENSION)[0]
                    stats_names.add(stat_name)
                    plt.figure(stat_name)
                    stat = np.load(os.path.join(stat_folder, stat_file))
                    if stat_name.startswith('classification'):
                        exp_classification_acc.append(stat[stat_index])
                    if stat_name.startswith('clustering_acc'):
                        exp_clustering_acc.append(stat[-1])
                    if args.plot_mean:
                        stats_dict[stat_name].append(stat)
                    else:
                        if j == 1:
                            label = str(i) if exp_labels is None else exp_labels[i]
                        else:
                            label = ''
                        color = colors[i]
                        plt.plot(range(len(stat)), stat, label=label, color=color)

        label = str(i) if exp_labels is None else exp_labels[i]
        exps_classification_ste += exp_classification_acc
        exps_clustering_ste += exp_clustering_acc
        exp_classification_acc = np.array(exp_classification_acc) * 100
        exp_clustering_acc = np.array(exp_clustering_acc) * 100

        print("the mean, std and ste classification accuracy of partition {} are: "
              "{}, {}, {}".format(label, np.mean(exp_classification_acc), np.std(exp_classification_acc),
                                  sem(exp_classification_acc)))
        exps_mean_classification_acc.append(np.mean(exp_classification_acc))
        if len(exp_clustering_acc) != 0:
            print("the mean and std clustering accuracy of partition {} are: "
                  "{}, {}".format(label, np.mean(exp_clustering_acc), np.std(exp_clustering_acc)))
            exps_mean_clustering_acc.append(np.mean(exp_clustering_acc))

        if args.plot_mean:
            for stat_name, stat_list in stats_dict.items():
                max_len = max([len(stat) for stat in stat_list])
                stat_nan_list = []
                for stat in stat_list:
                    num_nan = max_len - len(stat)
                    if num_nan > 0:
                        stat = np.concatenate([stat, np.full(shape=num_nan, fill_value=np.nan)])
                    stat_nan_list.append(stat)

                plt.figure(stat_name)
                plt.errorbar(range(max_len), np.nanmean(stat_nan_list, axis=0), np.nanstd(stat_nan_list, axis=0),
                             label=label, color=colors[i], alpha=0.3)

    print("the mean and std classification accuracy over all partitions are: "
          "{}, {}".format(np.mean(exps_mean_classification_acc), np.std(exps_mean_classification_acc)))
    if len(exps_mean_clustering_acc) != 0:
        print("the mean and std clustering accuracy over all partitions are: "
              "{}, {}".format(np.mean(exps_mean_clustering_acc), np.std(exps_mean_clustering_acc)))
        print('experiments clustering ste is: {}'.format(sem(np.array(exps_clustering_ste) * 100)))
    print('experiments classification ste is: {}'.format(sem(np.array(exps_classification_ste) * 100)))
    for stat_name in stats_names:
        plt.figure(stat_name)
        plt.xlabel('epoch')
        plt.ylabel(stat_name)
        if stat_name.startswith('classification'):
            plt.title('{}-{:.2f}+/-{:.2f}'.format(stat_name, np.mean(exps_mean_classification_acc),
                                                  np.std(exps_mean_classification_acc)))
        elif stat_name.startswith('clustering_acc'):
            plt.title('{}-{:.2f}+/-{:.2f}'.format(stat_name, np.mean(exps_mean_clustering_acc),
                                                  np.std(exps_mean_clustering_acc)))
        else:
            plt.title(stat_name)
        plt.legend(loc='lower right', ncol=3)
        graph_path = '{}.png'.format(os.path.join(graphs_folder, stat_name))
        plt.savefig(graph_path)
        plt.close()


if __name__ == '__main__':
    main()
