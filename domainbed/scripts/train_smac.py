from argparse import ArgumentParser
import random

from ConfigSpace import Configuration, ConfigurationSpace, Float, Categorical
import numpy as np
import torch
from smac import MultiFidelityFacade, Scenario
from smac.utils.logging import get_logger

from domainbed import algorithms,datasets, hparams_registry
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader

logger = get_logger(__name__)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    print("WARNING: No GPU available, running on CPU. This may be slow.")

def train(loaders, algorithm, budget):
    minibatches_iterator = zip(*loaders)

    for step in range(int(budget)):
        minibatches = [(x.to(device), y.to(device)) for x, y in next(minibatches_iterator)]
        algorithm.update(minibatches)

def lodo_objective(infinite_loaders, fast_loaders, dataset, config: Configuration, budget: int):
    # For each domain, train a model on all other domains and return the validation accuracy averaged over the left-out domains.
    algorithm_class = algorithms.get_algorithm_class("ERMPlusPlus")
    hparams = hparams_registry.default_hparams("ERMPlusPlus", dataset)
    hparams.update(dict(config))

    val_accuracy = 0.0

    for i, val_loader in enumerate(fast_loaders):
        train_loaders = [l for j, l in enumerate(infinite_loaders) if j != i]
        algorithm = algorithm_class(dataset.input_shape, dataset.num_classes, len(infinite_loaders) - 1, hparams)
        algorithm.to(device)
        train(train_loaders, algorithm, budget)
        val_accuracy += misc.accuracy(algorithm, val_loader, None, device)

    val_accuracy /= len(fast_loaders)
    logger.info(f"[LODO] Validation accuracy: {val_accuracy:.4f} for config: {config}")
    return 1.0 - val_accuracy

def holdout_objective(train_loaders, val_loaders, dataset, config: Configuration, budget: int):
    # Train a model with the given parameters, then return the validation accuracy measured on the in-domain holdout sets.
    algorithm_class = algorithms.get_algorithm_class("ERMPlusPlus")
    hparams = hparams_registry.default_hparams("ERMPlusPlus", dataset)
    hparams.update(dict(config))
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes, len(dataset) - 2, hparams)
    algorithm.to(device)

    train(train_loaders, algorithm, budget)
    
    val_accuracy = 0.0

    for val_loader in val_loaders:
        val_accuracy += misc.accuracy(algorithm, val_loader, None, device)

    val_accuracy /= len(val_loaders)
    logger.info(f"[Holdout] Validation accuracy: {val_accuracy:.4f} for config: {config}")
    return 1.0 - val_accuracy

def get_loaders(dataset, args):
    in_splits = []
    out_splits = []

    for env_i, env in enumerate(dataset):
        if args.test_env == env_i:
            in_splits.append(env)
            out_splits.append(env)
        else:
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
        for i, env in enumerate(in_splits)
        if i != args.test_env]

    val_loaders = [FastDataLoader(
        dataset=env,
        batch_size=args.batch_size,
        num_workers=dataset.N_WORKERS)
        for i, env in enumerate(out_splits)
        if i != args.test_env]
    
    env_infinite_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=None,
        batch_size=args.batch_size,
        num_workers=dataset.N_WORKERS)
        for env in dataset]

    env_fast_loaders = [FastDataLoader(
        dataset=env,
        batch_size=args.batch_size,
        num_workers=dataset.N_WORKERS)
        for env in dataset]
    
    return train_loaders, val_loaders, env_infinite_loaders, env_fast_loaders

def main():
    parser = ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--test_env', type=int)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--min_steps', type=int, default=500)
    parser.add_argument('--max_steps', type=int, default=2000)
    parser.add_argument('--model_selection', type=str, choices=['lodo', 'holdout'], default='lodo')
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--max_hpo_trials', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
            [args.test_env], hparams_registry.default_hparams("ERMPlusPlus", args.dataset))
    else:
        raise NotImplementedError
    
    train_loaders, val_loaders, env_infinite_loaders, env_fast_loaders = get_loaders(dataset, args)

    config_space = ConfigurationSpace({
        "resnet_dropout": Categorical("resnet_dropout", [0.0, 0.1, 0.5]),
        "linear_lr": Float("linear_lr", bounds=(1e-5, 1e-3), log=True),
        "weight_decay": Float("weight_decay", bounds=(1e-6, 1e-2), log=True),
        "lr": Float("lr", bounds=(1e-5, 1e-3), log=True),
    })

    scenario = Scenario(
        config_space,
        name=f"{args.dataset}-{args.test_env}-{args.model_selection}",
        deterministic=True,
        n_trials=args.max_hpo_trials,
        min_budget=args.min_steps,
        max_budget=args.max_steps)

    if args.model_selection == 'lodo':
        tr = [l for i, l in enumerate(env_infinite_loaders) if i != args.test_env]
        val = [l for i, l in enumerate(env_fast_loaders) if i != args.test_env]
        objective_function = lambda config, seed, budget: lodo_objective(tr, val, dataset, config, budget)
    elif args.model_selection == 'holdout':
        objective_function = lambda config, seed, budget: holdout_objective(train_loaders, val_loaders, dataset, config, budget)
    
    smac = MultiFidelityFacade(scenario, objective_function)
    incumbent = smac.optimize()

    # Train a final model with the best hyperparameters
    print(f"Training final model with hyperparameters: {incumbent}")
    algorithm_class = algorithms.get_algorithm_class("ERMPlusPlus")
    hparams = hparams_registry.default_hparams("ERMPlusPlus", dataset)
    hparams.update(dict(incumbent))
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes, len(dataset) - 1, hparams)
    algorithm.to(device)

    env_loaders_train = [l for i, l in enumerate(env_infinite_loaders) if i != args.test_env]
    train(env_loaders_train, algorithm, args.max_steps)
    test_accuracy = misc.accuracy(algorithm, env_fast_loaders[args.test_env], None, device)
    print(f"Test accuracy on environment {args.test_env}: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()
