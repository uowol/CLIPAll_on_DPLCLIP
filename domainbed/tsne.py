# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import collections
import json
import os
import random
import sys
import time
import uuid
from itertools import chain

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader, DataParallelPassthrough
from domainbed import model_selection
from domainbed.lib.query import Q

from datetime import datetime, timedelta

from sklearn.manifold import TSNE
import numpy as np
from matplotlib import pyplot as plt

import clip

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str, default="RotatedMNIST")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--task', type=str, default="domain_generalization",
        help='domain_generalization | domain_adaptation')
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0)
    parser.add_argument('--algorithm_path', type=str, default="/data4/kchanwo/clipall/train_results/CLIPALL_ViTB16_VLCS_T0/model.pkl")
    
    # parser.add_argument('--clip_backbone', type=str, default="None")
    args = parser.parse_args()

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0
    algorithm_path = args.algorithm_path
    algorithm_dict = torch.load(algorithm_path)['model_dict']
    if 'DPLCLIPALL' in args.algorithm:
        algorithm_dict['network.input.weight'] = algorithm_dict.pop('network.module.input.weight')
        algorithm_dict['network.input.bias'] = algorithm_dict.pop('network.module.input.bias')
        algorithm_dict['network.hiddens.0.weight'] = algorithm_dict.pop('network.module.hiddens.0.weight')
        algorithm_dict['network.hiddens.0.bias'] = algorithm_dict.pop('network.module.hiddens.0.bias')
        algorithm_dict['network.output.weight'] = algorithm_dict.pop('network.module.output.weight')
        algorithm_dict['network.output.bias'] = algorithm_dict.pop('network.module.output.bias')
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(args.output_dir)

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))

    hparams['test_envs'] = [int(i) for i in args.test_envs]

    hparams['clip_transform'] = hparams['backbone'] == 'clip'

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir, args.test_envs, hparams)
    else:
        raise NotImplementedError

    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.
    
    # To allow unsupervised domain adaptation experiments, we split each test
    # env into 'in-split', 'uda-split' and 'out-split'. The 'in-split' is used
    # by collect_results.py to compute classification accuracies.  The
    # 'out-split' is used by the Oracle model selectino method. The unlabeled
    # samples in 'uda-split' are passed to the algorithm at training time if
    # args.task == "domain_adaptation". If we are interested in comparing
    # domain generalization and domain adaptation results, then domain
    # generalization algorithms should create the same 'uda-splits', which will
    # be discared at training.
    in_splits = []
    out_splits = []
    uda_splits = []
    for env_i, env in enumerate(dataset):
        uda = []

        out, in_ = misc.split_dataset(env,
            int(len(env) * args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i))

        if env_i in args.test_envs:
            uda, in_ = misc.split_dataset(in_,
                int(len(in_)*args.uda_holdout_fraction),
                misc.seed_hash(args.trial_seed, env_i))

        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
            if uda is not None:
                uda_weights = misc.make_weights_for_balanced_classes(uda)
        else:
            in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))

        if len(uda):
            uda_splits.append((uda, uda_weights))

    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(in_splits)
        if i not in args.test_envs]
    
    uda_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(uda_splits)
        if i in args.test_envs]

    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=64,
        num_workers=dataset.N_WORKERS)
        for env, _ in (in_splits + out_splits + uda_splits)]
    eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
    eval_loader_names = ['env{}_in'.format(i)
        for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i)
        for i in range(len(out_splits))]
    eval_loader_names += ['env{}_uda'.format(i)
        for i in range(len(uda_splits))]

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
        len(dataset) - len(args.test_envs), hparams)

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)
        # print(algorithm.visual_projection)

    algorithm.to(device)
    if hasattr(algorithm, 'network'):
        algorithm.network = DataParallelPassthrough(algorithm.network)
    else:
        for m in algorithm.children():
            m = DataParallelPassthrough(m)

    train_minibatches_iterator = zip(*train_loaders)
    uda_minibatches_iterator = zip(*uda_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    now = datetime.now()
    now = now + timedelta(hours=9)

    print("LOG:",f"start visualization. time is \t{now.strftime('%Y-%m-%d %H:%M:%S')}.")

    clip_model = clip.load(hparams['clip_backbone'])[0].float()
    evals = zip(eval_loader_names, eval_loaders, eval_weights)
    eval_num = 0
    for name, loader, weights in evals:
        actual = []
        deep_features = [[] for _ in range(12)]
        deep_feature = []
        algorithm.eval()
        with torch.no_grad():
            for (x, y), _, __ in loader:
                x = x.to(device)
                y = y.to(device)

                image_feature = clip_model.encode_image(x)
                image_weight = algorithm.visual_network(image_feature)
                mean_image_weight = image_weight.mean(dim=0, keepdim=True)

                features = algorithm.encode_image(x)
                image_feature = torch.einsum('da,abc->bc', mean_image_weight, features)
                deep_feature += image_feature.cpu().numpy().tolist()
                actual += y.cpu().numpy().tolist()
                for k in range(12):
                    deep_features[k] += features[k].cpu().numpy().tolist()
        
        for k in range(12):
            tsne = TSNE(n_components=2, random_state=0)
            cluster = np.array(tsne.fit_transform(np.array(deep_features[k])))
            actual = np.array(actual)
            plt.figure(figsize=(10, 10))
            for i, label in zip(range(len(hparams['class_names'])), hparams['class_names']):
                idx = np.where(actual == i)
                plt.scatter(cluster[idx, 0], cluster[idx, 1], marker='.', label=label)
            
            plt.legend()
            plt.savefig(f'{args.output_dir}/tsne_eval{eval_num}_layer{k}.png', bbox_inches='tight')
            plt.close()

        tsne = TSNE(n_components=2, random_state=0)
        cluster = np.array(tsne.fit_transform(np.array(deep_feature)))
        actual = np.array(actual)
        plt.figure(figsize=(10, 10))
        for i, label in zip(range(len(hparams['class_names'])), hparams['class_names']):
            idx = np.where(actual == i)
            plt.scatter(cluster[idx, 0], cluster[idx, 1], marker='.', label=label)
        
        plt.legend()
        plt.savefig(f'{args.output_dir}/tsne_eval{eval_num}_weighted_sum.png', bbox_inches='tight')
        plt.close()

        eval_num += 1