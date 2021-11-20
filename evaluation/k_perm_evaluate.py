import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from options import MultiEvaluationOptions
import glob
from collections import defaultdict
import numpy as np
from evaluation.evaluate import main as eval
from xlwt import Workbook


MODEL_FILE = 'checkpoints/ckpt-{}{}.pkl'
SAVE_FOLDER = 'xl_files'


def main():
    args = MultiEvaluationOptions().parse()
    experiments = args.experiments.split(',')
    accs = defaultdict(list)
    eval_args = dict(vars(args))
    end = 'end_' if args.end_iter else ''
    phase = 'us' if args.us else 's'
    model_file = MODEL_FILE.format(end, phase)
    for i, exp in enumerate(experiments):
        exp_folder = glob.glob('{}*'.format(os.path.join(args.base_folder, exp)))[0]
        sub_exps = sorted(glob.glob('{}/[0-9]*'.format(exp_folder)))
        model_paths = [os.path.join(sub_exp, model_file) for sub_exp in sub_exps] if len(sub_exps) > 0 \
            else [os.path.join(exp_folder, model_file)]
        for k in args.ks:
            top_k_acc = []
            for model_path in model_paths:
                eval_args.update({'ckpt': model_path, 'top_k': k})
                model_top_k_acc = eval(extra_args=eval_args)
                top_k_acc.append(model_top_k_acc)
            accs[k].append((np.mean(top_k_acc), np.std(top_k_acc)))

    if not os.path.exists(SAVE_FOLDER):
        os.mkdir(SAVE_FOLDER)

    wb = Workbook()
    sheet1 = wb.add_sheet('results')
    sheet1.write(0, 0, 'k/partition')
    sheet2 = wb.add_sheet('no_std')
    sheet2.write(0, 0, 'k/partition')

    for i, (k, acc) in enumerate(accs.items(), 1):
        sheet1.write(i, 0, k)
        sheet2.write(i, 0, k)
        # print('k={}:\n\t'.format(k), end='')
        for j, (mean_acc, std_acc) in enumerate(acc, 1):
            if i == 1:
                sheet1.write(0, j, j)
                sheet2.write(0, j, j)
            sheet1.write(i, j, '{}\u00B1{}'.format(round(mean_acc * 100, 2), round(std_acc * 100, 2)))
            sheet2.write(i, j, round(mean_acc * 100, 2))
        wb.save(os.path.join(SAVE_FOLDER, '{}.xls'.format(args.file_name)))


if __name__ == '__main__':
    main()
