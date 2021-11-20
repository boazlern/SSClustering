import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image
from options import ShowGraphsOptions
import glob

EXPERIMENT_FOLDERS = ['experiments/cifar10', 'experiments/cifar100', 'experiments']


args = ShowGraphsOptions().parse()

experiment_folders = EXPERIMENT_FOLDERS if args.base_folder is None else [args.base_folder]
exp_folder = glob.glob('{}*'.format(os.path.join(experiment_folders[0], args.exp_id)))
i = 1
while len(exp_folder) == 0:
    exp_folder = glob.glob('{}*'.format(os.path.join(experiment_folders[i], args.exp_id)))
    i += 1

exp_folder = exp_folder[0]
sub_exps = sorted(glob.glob('{}/[0-9]*'.format(exp_folder)))
if len(sub_exps) == 0:
    graphs_folders = [os.path.join(exp_folder, 'graphs')] if args.base_folder is None else [exp_folder]
elif args.sub_exp is not None:
    graphs_folders = [os.path.join(exp_folder, args.sub_exp, 'graphs')]
else:
    graphs_folders = [os.path.join(sub_exps[i], 'graphs') for i in range(len(sub_exps))]

graphs = [[Image.open(os.path.join(graphs_folder, x)) for x in os.listdir(graphs_folder)]
          for graphs_folder in graphs_folders]

total_width = 0
line_heights = []

for sub_exp_graphs in graphs:
    line_width = 0
    line_height = 0
    for graph in sub_exp_graphs:
        w, h = graph.size
        line_width += w
        line_height = max(line_height, h)
    total_width = max(total_width, line_width)
    line_heights.append(line_height)

new_im = Image.new('RGB', (total_width, sum(line_heights)))

x_offset = 0
y_offset = 0
for i, sub_exp_graphs in enumerate(graphs):
    for graph in sub_exp_graphs:
        new_im.paste(graph, (x_offset, y_offset))
        x_offset += graph.size[0]
    y_offset = sum(line_heights[:i + 1])
    x_offset = 0

new_im.show()
