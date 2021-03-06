import argparse
import logging
import os
import pdb
import random
import shutil
import sys
from timeit import default_timer as timer
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import average_precision_score, confusion_matrix
from tensorboardX import SummaryWriter
from torch import Tensor
from torch.backends import cudnn
from tqdm import tqdm

import evaluation
import models
from loss import hybrid_regr_loss
from torchsummary import summary

from models import generate_square_subsequent_mask
from utils.data_generator import DataGenerator
from utils.utilities import (create_logging, doa_labels, event_labels,
                             get_filename, logging_and_writer,
                             print_evaluation, to_torch)
import optuna
import pickle
from math import log2

from torch.nn import ReLU, LeakyReLU, GELU

## Hyper-parameters
################# Model #################
USE_TRANSFORMER = True

Model_SED = 'CRNN10'            # 'CRNN10' | 'VGG9'
Model_DOA = 'pretrained_CRNN10' # 'pretrained_CRNN10' | 'pretrained_VGG9'
model_pool_type = 'avg'         # 'max' | 'avg'
model_pool_size = (2, 2)
model_interp_ratio = 16

loss_type = 'MAE'
################# param #################
batch_size = 32
Max_epochs = 10
lr = 1e-3
weight_decay = 0
threshold = {'sed': 0.3}

fs = 32000
nfft = 1024
hopsize = 320 # 640 for 20 ms
mel_bins = 96
frames_per_1s = fs // hopsize
sub_frames_per_1s = 50
chunklen = int(2 * frames_per_1s)
hopframes = int(0.5 * frames_per_1s)

hdf5_folder_name = '{}fs_{}nfft_{}hs_{}melb'.format(fs, nfft, hopsize, mel_bins)
################# batch intervals for save & lr update #################
save_interval = 2000
lr_interval = 2000
########################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bptt = 1


def get_batch(source: Tensor, i: int) -> Tuple[Tensor, Tensor]:
    """
    Args:
        source: Tensor, shape [full_seq_len, batch_size]
        i: int

    Returns:
        tuple (data, target), where data has shape [seq_len, batch_size] and
        target has shape [seq_len * batch_size]
    """
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target


def train(args, data_generator, model, optimizer, logging):
    global optuna_idx

    # Set the writer
    writer = SummaryWriter()
    writer.add_text('Parameters', str(args))

    temp_submissions_dir = os.path.join(appendixes_dir, '__submissions__')

    trial = 0
    while os.path.isdir(os.path.join(temp_submissions_dir, 'trial_{}'.format(trial))):
        trial += 1

    temp_submissions_dir_train = os.path.join(temp_submissions_dir, 'trial_{}'.format(trial), 'train')
    temp_submissions_dir_valid = os.path.join(temp_submissions_dir, 'trial_{}'.format(trial), 'valid')

    logging.info('\n===> Training mode')

    train_begin_time = timer()
    src_mask = generate_square_subsequent_mask(bptt).to(device)


    epoch_size = data_generator.epoch_size

    iterator = tqdm(enumerate(data_generator.generate_train()),
        total=Max_epochs*epoch_size, unit='batch')

    for batch_idx, (batch_x, batch_y_dict) in iterator:

        epochs = int(batch_idx//epoch_size)
        epoch_batches = int(batch_idx%epoch_size)

        ################
        ## Validation
        ################
        if batch_idx % 200 == 0:

            valid_begin_time = timer()
            train_time = valid_begin_time - train_begin_time

            # Train evaluation
            shutil.rmtree(temp_submissions_dir_train, ignore_errors=True)
            os.makedirs(temp_submissions_dir_train, exist_ok=False)
            train_metrics = evaluation.evaluate(
                        data_generator=data_generator, 
                        data_type='train', 
                        max_audio_num=30,
                        task_type=args.task_type, 
                        model=model, 
                        cuda=args.cuda,
                        loss_type=loss_type,
                        threshold=threshold,
                        submissions_dir=temp_submissions_dir_train, 
                        frames_per_1s=frames_per_1s,
                        sub_frames_per_1s=sub_frames_per_1s)

            logging.info('----------------------------------------------------------------------------------------------------------------------------------------------')
            
            # Validation evaluation
            if args.fold != -1:
                shutil.rmtree(temp_submissions_dir_valid, ignore_errors=True)
                os.makedirs(temp_submissions_dir_valid, exist_ok=False)
                valid_metrics = evaluation.evaluate(
                        data_generator=data_generator, 
                        data_type='valid', 
                        max_audio_num=30,
                        task_type=args.task_type, 
                        model=model, 
                        cuda=args.cuda,
                        loss_type=loss_type,
                        threshold=threshold, 
                        submissions_dir=temp_submissions_dir_valid, 
                        frames_per_1s=frames_per_1s, 
                        sub_frames_per_1s=sub_frames_per_1s)
                metrics = [train_metrics, valid_metrics]
                logging_and_writer('valid', metrics, logging, writer, batch_idx)
            else:
                logging_and_writer('train', train_metrics, logging, writer, batch_idx)

            valid_time = timer() - valid_begin_time
            logging.info('Iters: {},  Epochs/Batches: {}/{},  Train time: {:.3f}s,  Eval time: {:.3f}s'.format(
                        batch_idx, epochs, epoch_batches, train_time, valid_time))             
            logging.info('----------------------------------------------------------------------------------------------------------------------------------------------')
            train_begin_time = timer()

        ###############
        ## Save model
        ###############
        if batch_idx % save_interval == 0 and batch_idx > 1:
            save_subfolder = os.path.join(models_dir, str(optuna_idx))
            os.makedirs(save_subfolder, exist_ok=True)
            save_path = os.path.join(save_subfolder, 'iter_{}.pth'.format(batch_idx))
            checkpoint = {'model_state_dict': model.module.state_dict(),
                          'optimizer_state_dict': optimizer.state_dict(),
                          'rng_state': torch.get_rng_state(),
                          'cuda_rng_state': torch.cuda.get_rng_state()}
            torch.save(checkpoint, save_path)
            logging.info('Checkpoint saved to {}'.format(save_path))

        ###############
        ## Train
        ###############
        # Reduce learning rate
        if batch_idx % lr_interval == 0 and batch_idx > 1:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.9

        batch_x = to_torch(batch_x, args.cuda)
        batch_y_dict = {
            'events':   to_torch(batch_y_dict['events'], args.cuda),
            'doas':  to_torch(batch_y_dict['doas'], args.cuda)
        }

        # Forward
        model.train()
        if batch_size != bptt:  # only on last batch
            src_mask = src_mask[:batch_size, :batch_size]
        output = model(batch_x)
        
        # Loss
        seld_loss, _, _ = hybrid_regr_loss(output, batch_y_dict, args.task_type, loss_type=loss_type)

        # Backward
        optimizer.zero_grad()
        seld_loss.backward()
        optimizer.step()

        if batch_idx == Max_epochs*epoch_size:
            iterator.close()
            writer.close()
            break

    optuna_idx += 1

    return (1 - metrics[1][2][1])

def inference(args, data_generator, logging):
  
    # Load model for sed only
    print('\n===> Inference for SED')

    if args.task_type == 'two_staged_eval':

        model_path = os.path.join(models_dir, 'sed_only',
                                'model_' + Model_SED + '_{}'.format(args.audio_type) + '_fold_{}'.format(args.fold) + '_seed_{}'.format(args.seed),
                                'iter_{}.pth'.format(args.iteration))
        assert os.path.exists(model_path), 'Error: no checkpoint file found!'
        model = models.__dict__[Model_SED](class_num, args.model_pool_type,
                                           args.model_pool_size, args.model_interp_ratio, pretrained_path, use_transformer=USE_TRANSFORMER)
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['model_state_dict'])
        if args.cuda:
            model.cuda()

        fold_submissions_dir= os.path.join(submissions_dir, args.task_type, 'model_' + args.model + '_{}'.format(args.audio_type) + \
            '_seed_{}'.format(args.seed), '_test')
        shutil.rmtree(fold_submissions_dir, ignore_errors=True)
        os.makedirs(fold_submissions_dir, exist_ok=False)
        test_metrics = evaluation.evaluate(
                data_generator=data_generator, 
                data_type='test', 
                max_audio_num=None,
                task_type='sed_only', 
                model=model, 
                cuda=args.cuda,
                loss_type=loss_type,
                threshold=threshold,
                submissions_dir=fold_submissions_dir, 
                frames_per_1s=frames_per_1s,
                sub_frames_per_1s=sub_frames_per_1s)
        
        # Load model for doa using sed pred
        print('\n===> Inference for SED and DOA')
        model_path = os.path.join(models_dir, 'doa_only',
                                'model_' + Model_DOA + '_{}'.format(args.audio_type) + '_fold_{}'.format(args.fold) + '_seed_{}'.format(args.seed),
                                'iter_{}.pth'.format(args.iteration))
        assert os.path.exists(model_path), 'Error: no checkpoint file found!'
        model = models.__dict__[Model_DOA](class_num, args.model_pool_type, 
            args.model_pool_size, args.model_interp_ratio, pretrained_path, use_transformer=USE_TRANSFORMER)
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['model_state_dict'])
        if args.cuda:
            model.cuda()

        test_metrics = evaluation.evaluate(
                data_generator=data_generator, 
                data_type='test', 
                max_audio_num=None,
                task_type='two_staged_eval', 
                model=model, 
                cuda=args.cuda,
                loss_type=loss_type,
                threshold=threshold,
                submissions_dir=fold_submissions_dir, 
                frames_per_1s=frames_per_1s,
                sub_frames_per_1s=sub_frames_per_1s)

    else:
        # 'sed_only' | 'doa_only' | 'seld' 
        model_path = os.path.join(models_dir, args.task_type,
                                'model_' + args.model + '_{}'.format(args.audio_type) + '_fold_{}'.format(args.fold) + '_seed_{}'.format(args.seed),
                                'iter_{}.pth'.format(args.iteration))
        assert os.path.exists(model_path), 'Error: no checkpoint file found!'
        model = models.__dict__[args.model](class_num, args.model_pool_type, 
            args.model_pool_size, args.model_interp_ratio, pretrained_path)
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['model_state_dict'])
        if args.cuda:
            model.cuda()

        fold_submissions_dir= os.path.join(submissions_dir, args.task_type, 'model_' + args.model + '_{}'.format(args.audio_type) + \
            '_seed_{}'.format(args.seed), '_test')
        shutil.rmtree(fold_submissions_dir, ignore_errors=True)
        os.makedirs(fold_submissions_dir, exist_ok=False)
        test_metrics = evaluation.evaluate(
                data_generator=data_generator, 
                data_type='test', 
                max_audio_num=None,
                task_type=args.task_type, 
                model=model, 
                cuda=args.cuda,
                loss_type=loss_type,
                threshold=threshold,
                submissions_dir=fold_submissions_dir, 
                frames_per_1s=frames_per_1s,
                sub_frames_per_1s=sub_frames_per_1s)

    logging.info('----------------------------------------------------------------------------------------------------------------------------------------------')
    logging_and_writer('test', test_metrics, logging)
    logging.info('----------------------------------------------------------------------------------------------------------------------------------------------')

    test_submissions_dir= os.path.join(submissions_dir, args.task_type, 'model_' + args.model + '_{}'.format(args.audio_type) + \
        '_seed_{}'.format(args.seed), 'test')
    os.makedirs(test_submissions_dir, exist_ok=True)
    for fn in sorted(os.listdir(fold_submissions_dir)):
        if fn.endswith('.csv') and not fn.startswith('.'):
            src = os.path.join(fold_submissions_dir, fn)
            dst = os.path.join(test_submissions_dir, fn)
            shutil.copyfile(src, dst)


def inference_all_fold(args):

    test_submissions_dir= os.path.join(args.workspace, 'appendixes', 'submissions', args.task_type,
        'model_' + args.model  + '_{}'.format(args.audio_type) + \
            '_seed_{}'.format(args.seed), 'test')    

    gt_meta_dir = '/home/username/deep_learning_project/datasets/extracted/metadata_dev/'
    sed_scores, doa_er_metric, seld_metric = evaluation.calculate_SELD_metrics(gt_meta_dir, test_submissions_dir, score_type='all')

    loss = [0.0, 0.0, 0.0]
    sed_mAP = [0.0, 0.0]

    metrics = [loss, sed_mAP, sed_scores, doa_er_metric, seld_metric]

    print('----------------------------------------------------------------------------------------------------------------------------------------------')
    print_evaluation(metrics)
    print('----------------------------------------------------------------------------------------------------------------------------------------------')


def reset_model():
    global models_dir
    if args.mode == 'train':
        # models directory
        models_dir = os.path.join(appendixes_dir, 'models_saved', '{}'.format(args.task_type),
                                  'model_' + args.model + '_{}'.format(args.audio_type) + '_fold_{}'.format(args.fold) +
                                  '_seed_{}'.format(args.seed))
        os.makedirs(models_dir, exist_ok=True)
    elif args.mode == 'inference':
        # models directory
        models_dir = os.path.join(appendixes_dir, 'models_saved')

    logging.info('\n===> Building model')
    train_model = models.__dict__[args.model](class_num, args.model_pool_type,
                                        args.model_pool_size, args.model_interp_ratio, pretrained_path)

    if args.cuda:
        logging.info('\nUtilize GPUs for computation')
        logging.info('\nNumber of GPU available: {}'.format(torch.cuda.device_count()))
        if torch.cuda.device_count() > 1:
            Multi_GPU = True
        else:
            Multi_GPU = False
        train_model.cuda()
        # cudnn.benchmark = False # for cuda 10.0
        train_model = torch.nn.DataParallel(train_model)

    return train_model


def objective(trial, arguments):
    train_model = reset_model()

    lr = trial.suggest_float('lr', 1e-4, 1e-3)
    nhead = trial.suggest_int('nhead', 1, 5)
    optimzer_name = trial.suggest_categorical('optimzer', ['SGD', 'Adam', 'RMSprop'])
    activations = {"relu": ReLU(), "gelu": GELU(), "leaky_relu": LeakyReLU()}
    activation = trial.suggest_categorical('activation', list(activations.keys()))
    logging.info(f"lr: {lr}, optimizer: {optimzer_name}, activation: {activation}, nhead: {2 ** nhead}")

    # lstm_layers = trial.suggest_int('lstm_layers', 1, 2)
    # logging.info(f"lr: {lr}, optimizer: {optimzer_name}, layers: {lstm_layers}")
    # train_model.module.init_weights(lstm_layers)

    if optimzer_name == 'Adam':
        optimizer = optim.Adam(train_model.parameters(), lr=lr,
                               betas=(0.9, 0.999), eps=1e-08,
                               weight_decay=weight_decay, amsgrad=True)
    elif optimzer_name == 'SGD':
        optimizer = optim.SGD(train_model.parameters(), lr=lr, weight_decay=weight_decay, )
    elif optimzer_name == 'RMSprop':
        optimizer = optim.RMSprop(train_model.parameters(), lr=lr, weight_decay=weight_decay, )

    train_model.module.change_encoder(activations[activation], nhead)
    train_model.module.to(torch.device('cuda:0'))

    parameter_num = sum([param.numel() for param in train_model.parameters()])
    logging.info('\nTotal number of parameters: {}\n'.format(parameter_num))

    return train(arguments, data_generator, train_model, optimizer, logging)


if __name__ == '__main__':
    # global optuna_idx
    # optuna_idx = 0

    parser = argparse.ArgumentParser(description='DCASE2019 task3')

    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--workspace', type=str, required=True,
                                help='workspace directory')
    parser_train.add_argument('--feature_dir', type=str, required=True,
                                help='feature directory')
    parser_train.add_argument('--feature_type', type=str, required=True,
                                choices=['logmel', 'logmelgcc'])
    parser_train.add_argument('--audio_type', type=str, required=True, 
                              choices=['foa', 'mic'], help='audio type')
    parser_train.add_argument('--task_type', type=str, required=True,
                                choices=['sed_only', 'doa_only', 'two_staged_eval', 'seld'])
    parser_train.add_argument('--fold', default=1, type=int,
                                help='fold for cross validation, if -1, use full data')
    parser_train.add_argument('--seed', default='42', type=int,
                                help='random seed')
    parser_train.add_argument('--optuna', default='0', type=int,
                              help='optuna train index')

    parser_inference = subparsers.add_parser('inference')
    parser_inference.add_argument('--workspace', type=str, required=True,
                                help='workspace directory')
    parser_inference.add_argument('--feature_dir', type=str, required=True,
                                help='feature directory')
    parser_inference.add_argument('--feature_type', type=str, required=True,
                                choices=['logmel', 'logmelgcc'])
    parser_inference.add_argument('--audio_type', type=str, required=True, 
                                choices=['foa', 'mic'], help='audio type')
    parser_inference.add_argument('--task_type', type=str, required=True,
                                choices=['sed_only', 'doa_only', 'two_staged_eval', 'seld'])
    parser_inference.add_argument('--fold', default=1, type=int,
                                help='fold for cross validation, if -1, use full data')
    parser_inference.add_argument('--iteration', default=5000, type=int,
                                help='which iteration model to read')                                
    parser_inference.add_argument('--seed', default='42', type=int,
                                help='random seed')  

    parser_inference_all = subparsers.add_parser('inference_all')
    parser_inference_all.add_argument('--workspace', type=str, required=True,
                                help='workspace directory')
    parser_inference_all.add_argument('--audio_type', type=str, required=True, 
                                choices=['foa', 'mic'], help='audio type')
    parser_inference_all.add_argument('--task_type', type=str, required=True,
                                choices=['sed_only', 'doa_only', 'two_staged_eval', 'seld'])
    parser_inference_all.add_argument('--seed', default='42', type=int,
                                help='random seed')  

    args = parser.parse_args()



    '''
    1. Miscellaneous
    '''
    global optuna_idx
    optuna_idx = args.optuna
    print('Optuna index: ', optuna_idx)

    args.fs = fs
    args.nfft = nfft
    args.hopsize = hopsize
    args.mel_bins = mel_bins
    args.chunklen = chunklen
    args.hopframes = hopframes

    args.cuda = torch.cuda.is_available()
    args.batch_size = batch_size
    args.lr = lr
    args.weight_decay = weight_decay
    args.hdf5 = hdf5_folder_name

    if args.task_type == 'sed_only' or args.task_type == 'seld':
        args.model = Model_SED
    elif args.task_type == 'doa_only' or args.task_type == 'two_staged_eval':
        args.model = Model_DOA
    args.model_pool_type = model_pool_type
    args.model_pool_size = model_pool_size
    args.model_interp_ratio = model_interp_ratio
    args.loss_type = loss_type

    class_num = len(event_labels)
    doa_num = len(doa_labels)

    # inference all folds, otherwise train or inference single fold
    if args.mode == 'inference_all':
        inference_all_fold(args)
        sys.exit()

    # Get reproducible results by manually seed the random number generator
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cudnn.deterministic=True

    # logs directory
    logs_dir = os.path.join(args.workspace, 'logs', args.task_type, args.mode, 
            'model_' + args.model + '_{}'.format(args.audio_type) + '_fold_{}'.format(args.fold) + 
            '_seed_{}'.format(args.seed))
    create_logging(logs_dir, filemode='w')
    logging.info(args)

    # appendixes directory
    global appendixes_dir
    appendixes_dir = os.path.join(args.workspace, 'appendixes')
    os.makedirs(appendixes_dir, exist_ok=True)

    # submissions directory
    global submissions_dir
    submissions_dir = os.path.join(appendixes_dir, 'submissions')
    os.makedirs(submissions_dir, exist_ok=True)

    # pretrained path
    global pretrained_path
    # base_path = '/home/username/path'
    # pretrained_path = os.path.join(base_path, str(optuna_idx), 'iter_10000.pth')  # Take only the 10k saved step.

    pretrained_path = '/home/username/workspace/appendixes/models_saved/sed_only/model_CRNN10_mic_fold_2_seed_20/5/iter_10000.pth'

    '''
    2. Model
    '''
    model = reset_model()

    # global models_dir
    # if args.mode == 'train':
    #     # models directory
    #     models_dir = os.path.join(appendixes_dir, 'models_saved', '{}'.format(args.task_type),
    #                             'model_' + args.model + '_{}'.format(args.audio_type) + '_fold_{}'.format(args.fold) +
    #                             '_seed_{}'.format(args.seed))
    #     os.makedirs(models_dir, exist_ok=True)
    # elif args.mode == 'inference':
    #     # models directory
    #     models_dir = os.path.join(appendixes_dir, 'models_saved')
    #
    # logging.info('\n===> Building model')
    # model = models.__dict__[args.model](class_num, args.model_pool_type,
    #     args.model_pool_size, args.model_interp_ratio, pretrained_path)
    #
    #
    # if args.cuda:
    #     logging.info('\nUtilize GPUs for computation')
    #     logging.info('\nNumber of GPU available: {}'.format(torch.cuda.device_count()))
    #     if torch.cuda.device_count() > 1:
    #         Multi_GPU = True
    #     else:
    #         Multi_GPU = False
    #     model.cuda()
    #     # cudnn.benchmark = False # for cuda 10.0
    #     model = torch.nn.DataParallel(model)

    # Print the model architecture and parameters
    # logging.info('\nModel architectures:\n{}\n'.format(model))
    # logging.info('\nParameters and size:')
    # for n, (name, param) in enumerate(model.named_parameters()):
    #     logging.info('{}: {}'.format(name, list(param.size())))
    # parameter_num = sum([param.numel() for param in model.parameters()])
    # logging.info('\nTotal number of parameters: {}\n'.format(parameter_num))

    '''
    3. Data generator
    '''
    hdf5_dir = os.path.join(args.feature_dir, args.feature_type,
                             hdf5_folder_name, args.audio_type)
    data_generator = DataGenerator(
        args=args,
        hdf5_dir=hdf5_dir,
        logging=logging
    )

    '''
    4. Train, test and evaluation
    '''

    if args.mode == 'train':
        if args.task_type == 'sed_only':
            study = optuna.create_study()
            study.optimize(lambda trial: objective(trial, args), n_trials=10)
            logging.info(study.best_params)

            # optimizer = optim.Adam(model.parameters(), lr=lr,
            #                        betas=(0.9, 0.999), eps=1e-08,
            #                        weight_decay=weight_decay, amsgrad=True)
            # model.module.change_encoder(LeakyReLU(), 4)
            # model.module.to(torch.device('cuda:0'))
            # train(args, data_generator, model, optimizer, logging)

        elif args.task_type == 'doa_only':
            # params_file_path = '/home/lzpz6s/deep_optuna/sed_doae/logs/sed_only/train/model_CRNN10_mic_fold_2_seed_20/hyperparams_SED.pickle'
            # with open(params_file_path, 'rb') as handle:
            #     hyperparams = pickle.load(handle)
            #
            # lr = hyperparams[optuna_idx]['lr']
            # # nhead = int(log2(hyperparams[optuna_idx]['nhead']))
            # # activations = {"relu": ReLU(), "gelu": GELU(), "leaky_relu": LeakyReLU()}
            # # activation = hyperparams[optuna_idx]['activation']
            # lstm_layers = hyperparams[optuna_idx]['lstm_layers']
            # optimzer_name = hyperparams[optuna_idx]['optimizer']
            # if optimzer_name == 'Adam':
            #     optimizer = optim.Adam(model.parameters(), lr=lr,
            #                            betas=(0.9, 0.999), eps=1e-08,
            #                            weight_decay=weight_decay, amsgrad=True)
            # elif optimzer_name == 'SGD':
            #     optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, )
            # elif optimzer_name == 'RMSprop':
            #     optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay, )
            #
            # # model.module.change_encoder(activations[activation], nhead)
            # model.module.init_weights(lstm_layers)
            # model.to(torch.device('cuda:0'))
            # # logging.info(f"lr: {lr}, optimizer: {optimzer_name}, activation: {activation}, nhead: {2 ** nhead}")
            # logging.info(f"lr: {lr}, optimizer: {optimzer_name}, layers: {lstm_layers}")
            # train(args, data_generator, model, optimizer, logging)

            optimizer = optim.Adam(model.parameters(), lr=lr,
                                   betas=(0.9, 0.999), eps=1e-08,
                                   weight_decay=weight_decay, amsgrad=True)
            model.module.change_encoder(LeakyReLU(), 4)
            model.module.to(torch.device('cuda:0'))
            train(args, data_generator, model, optimizer, logging)

    elif args.mode == 'inference':
        inference(args, data_generator, logging)
    else:
        raise Exception('Error!')
