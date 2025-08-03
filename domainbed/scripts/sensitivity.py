from argparse import ArgumentParser
import random

import numpy as np
import torch

from domainbed import algorithms,datasets, hparams_registry
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader

def get_loaders(dataset, args):
    in_splits = []
    out_splits = []

    for env_i, env in enumerate(dataset):
        out, in_ = misc.split_dataset(env,
            int(len(env)*args.holdout_fraction),
            misc.seed_hash(args.seed, env_i))

        in_splits.append(in_)
        out_splits.append(out)

    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=None,
        batch_size=args.batch_size,
        num_workers=dataset.N_WORKERS)
        for i, env in enumerate(in_splits)]

    val_loaders = [FastDataLoader(
        dataset=env,
        batch_size=args.batch_size,
        num_workers=dataset.N_WORKERS)
        for i, env in enumerate(out_splits)]
    
    return train_loaders, val_loaders

def main():
    parser = ArgumentParser(description="Run experiments that measure sensitivity to hyperparameters using both in-domain and out-of-domain data.")
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing datasets')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation')
    parser.add_argument('--holdout_fraction', type=float, default=0.2, help='Fraction of data to hold out for validation')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--algorithm', type=str, choices=["RES"], default="RES", help='Algorithm to measure the sensitivity of hyperparameters')
    parser.add_argument('--max_steps', type=int, default=2000, help='Maximum number of training steps per environment')
    args = parser.parse_args()

    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset in vars(datasets):
        dataset_class = vars(datasets)[args.dataset]
        dataset = dataset_class(args.data_dir, dataset_class.ENVIRONMENTS, hparams_registry.default_hparams(args.algorithm, args.dataset))
    else:
        raise ValueError(f"Dataset {args.dataset} not found in datasets module.")

    train_loaders, val_loaders = get_loaders(dataset, args)

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)

    if args.algorithm == "RES":
        hparam_name = "decay_stu"
        hparam_values = [0.0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    else:
        raise ValueError(f"Algorithm {args.algorithm} not supported for sensitivity analysis.")

    print(f"{hparam_name},val_env,ood_acc,id_acc,ood_acc_n,id_acc_n")

    for hparam_value in hparam_values:
        hparams[hparam_name] = hparam_value

        for i, val_loader in enumerate(val_loaders):
            tr = [l for j, l in enumerate(train_loaders) if j != i]
            minibatches_iterator = zip(*tr)
            algorithm = algorithm_class(dataset.input_shape, dataset.num_classes, len(tr), hparams)
            algorithm.to(device)
            
            for step in range(args.max_steps):
                minibatches = [(x.to(device), y.to(device)) for x, y in next(minibatches_iterator)]
                algorithm.update(minibatches)
            
            ood_acc = misc.accuracy(algorithm, val_loader, None, device)
            ood_acc_n = len(val_loader.dataset)
            id_acc = np.mean([misc.accuracy(algorithm, train_loader, None, device) for j, train_loader in enumerate(train_loaders) if j != i])
            id_acc_n = sum([len(train_loader.dataset) for j, train_loader in enumerate(train_loaders) if j != i])
            print(f"{hparam_value},{i},{ood_acc:.4f},{id_acc:.4f},{ood_acc_n},{id_acc_n}")

if __name__ == '__main__':
    main()
