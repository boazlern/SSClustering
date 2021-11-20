import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from options import MultiEvaluationOptions
import glob
from evaluation.evaluate import main as eval


MODEL_FILE = 'checkpoints/ckpt-{}{}.pkl'
SAVE_FOLDER = 'xl_files'


def main():
    args = MultiEvaluationOptions().parse()
    args.runs_ensemble = True
    experiments = args.experiments.split(',')
    eval_args = dict(vars(args))
    end = 'end_' if args.end_iter else ''
    phase = 'us' if args.us else 's'
    model_file = MODEL_FILE.format(end, phase)
    for i, exp in enumerate(experiments, 1):
        exp_folder = glob.glob('{}*'.format(os.path.join(args.base_folder, exp)))[0]
        sub_exps = sorted(glob.glob('{}/[0-9]*'.format(exp_folder)))
        model_paths = [os.path.join(sub_exp, model_file) for sub_exp in sub_exps] if len(sub_exps) > 0 \
            else [os.path.join(exp_folder, model_file)]
        eval_args.update({'ckpt': model_paths})
        ensemble_acc = eval(extra_args=eval_args)
        print('experiment number {} got ensemble accuracy of: {}'.format(i, ensemble_acc))


if __name__ == '__main__':
    main()
