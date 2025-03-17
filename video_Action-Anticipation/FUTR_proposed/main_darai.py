import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
import pdb
import random
from torch.backends import cudnn
from opts import parser
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

#from loss.spc import SupConLoss#SupervisedContrastiveLoss
from utils import read_mapping_dict
from data.basedataset_darai import BaseDataset
#from data.basedataset_darai_depth import BaseDataset
#from data.basedataset_darai_gaze import BaseDataset
#from data.basedataset_darai_llm import BaseDataset
#from data.basedataset_darai_llm_train import BaseDataset
#from model.futr_proposed import FUTR
from model.futr_baseline import FUTR
#from model.rnn import ModelRNN
#from model.futr_safuser_tokenfusion import FUTR
#from model.rnn import FUTR
#from model.cnn import FUTR
#from model.afft import FUTR
#from model.futr_safuser_batchnormalization import FUTR
#from model.futr_safuser_depth import FUTR
#from model.futr_unsupervised_depth import FUTR
#from model.futr_unsupervised import FUTR
#from model.futr_unsupervised_llm import FUTR
#from model.futr_unsupervised_multimodal import FUTR
#from model.futr_unsupervised_temp2 import FUTR
#from model.futr_unsupervised_temp3 import FUTR
#from model.futr_unsupervised_temp4 import FUTR
from train_proposed import train
#from train_proposed_depth import train
#from train_unsupervised import train
#from train_llm import train
#from predict_darai_temp2 import predict
#from predict import predict
#from predict_darai import predict
from make_gif import predict
#from make_gif_llm import predict
#from make_gif_with_query import predict
# import gc

# torch.cuda.empty_cache()
# gc.collect()
device = torch.device('cuda')

def main(seed, dataset_ops):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark, cudnn.deterministic = False, True
    args = parser.parse_args()

    if args.cpu:
        device = torch.device('cpu')
        print('using cpu')
    else:
        device = torch.device('cuda')
        print('using gpu')
    
    print('runs : ', args.runs)
    print('model type : ', args.model)
    print('input type : ', args.input_type)
    print('Epoch : ', args.epochs)
    print("batch size : ", args.batch_size)
    print("Split : ", args.split)

    dataset = args.dataset
    task = args.task
    split = args.split

    if dataset == 'breakfast':
        data_path = './datasets/breakfast'
    elif dataset == '50salads' :
        data_path = './datasets/50salads'
    elif dataset == 'darai':
        data_path = './datasets/darai'

    query_mapping_file = os.path.join(data_path, 'mapping_l3_changed.txt')
    query_dict = read_mapping_dict(query_mapping_file)

    mapping_file = os.path.join(data_path, 'mapping_l2_changed.txt') ################_l2_changed
    actions_dict = read_mapping_dict(mapping_file)
    video_file_path = os.path.join(data_path, 'splits', 'train_split.txt' )
    video_val_path = os.path.join(data_path, 'splits', 'val_split.txt')

    # video_file_path = os.path.join(data_path, 'splits', 'traintemp.txt' )
    # video_val_path = os.path.join(data_path, 'splits', 'valtemp.txt')
    video_file_test_path = os.path.join(data_path, 'splits', 'test_split.txt')

    video_file = open(video_file_path, 'r')
    video_file_test = open(video_file_test_path, 'r')

    video_list = video_file.read().split('\n')[:-1]
    video_test_list = video_file_test.read().split('\n')[:-1]

    features_path = os.path.join(data_path, 'features_img')
    depth_features_path = os.path.join(data_path, 'features_depth')
    gt_path = os.path.join(data_path, 'groundTruth_img')

    #actions_dict = query_dict

    n_class = len(actions_dict) + 1
    pad_idx = n_class + 1 #+1


    # Model specification
    #model = ModelRNN(2048, n_class, args.hidden_dim, args.max_pos_len, 2)
    model = FUTR(n_class, args.hidden_dim, device=device, args=args, src_pad_idx=pad_idx,
                            n_query=args.n_query, n_head=args.n_head,
                            num_encoder_layers=args.n_encoder_layer, num_decoder_layers=args.n_decoder_layer).to(device)

    model_save_path = os.path.join('./save_dir', args.dataset, args.task, 'model/transformer', split, args.input_type, \
                                    'runs'+str(args.runs), f'_{str(dataset_ops)}')
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    results_save_path = os.path.join('./save_dir/'+args.dataset+'/'+args.task+'/results/transformer', 'split'+split,
                                    args.input_type )
    if not os.path.exists(results_save_path):
        os.makedirs(results_save_path)


    model_save_file = os.path.join(model_save_path, 'checkpoint.ckpt')
    model = nn.DataParallel(model).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)
    warmup_epochs = args.warmup_epochs
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=warmup_epochs, max_epochs=args.epochs)
    criterion = nn.MSELoss(reduction = 'none')
    #finegrained_criterion = SupConLoss(temperature=args.temperature).to(device)


    if args.predict :
        obs_perc = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        results_save_path = results_save_path +'/runs'+ str(args.runs) +'.txt'
        if args.dataset == 'breakfast' :
            model_path = '/home/seulgi/work/darai-anticipation/FUTR_proposed/save_dir/breakfast/long/model/transformer/1/i3d_transcript/runs0_all_losschanged/checkpoint4.ckpt'#'./ckpt/bf_split'+args.split+'.ckpt'
            #model_path = '/home/seulgi/work/darai-anticipation/FUTR_proposed/save_dir/breakfast/long/model/transformer/1/i3d_transcript/runs0_queryall_trainonly/checkpoint59.ckpt'
        elif args.dataset == '50salads':
            model_path = './ckpt/50s_split'+args.split+'.ckpt'
        elif args.dataset == 'darai':
            model_path = f'/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/save_dir/darai/long/model/transformer/1/i3d_transcript/runs0/_{dataset_ops}/seed_{seed}_best.ckpt'
            
        print("Predict with ", model_path)
        seed_list =  [1]#, 10, 13452]
        
        for obs_p in obs_perc :
            ant_whole = 0
            seg_whole = 0
            for seed in seed_list:
                model_path = f'/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/save_dir/darai/long/model/transformer/1/i3d_transcript/runs0/_{dataset_ops}/seed_{seed}_best.ckpt'
                model.load_state_dict(torch.load(model_path))
                model.to(device)
                ant, seg = 0, 0
                predict(model, video_test_list, args, obs_p, n_class, actions_dict, device, query_dict)
                ant_whole += ant
                seg_whole += seg
            print("**********************", obs_p, ant_whole/3, seg_whole/3)
    else :
        # Training
        #trainset = BaseDataset(video_file_path, actions_dict, features_path, gt_path, pad_idx, n_class, n_query=args.n_query, args=args, query_dict=query_dict)

        trainset = BaseDataset(video_file_path, actions_dict, features_path, depth_features_path, gt_path, pad_idx, n_class, n_query=args.n_query, args=args, query_dict=query_dict)
        print(len(trainset))
        train_loader = DataLoader(trainset, batch_size=args.batch_size, \
                                                    shuffle=True, num_workers=args.workers,
                                                    collate_fn=trainset.my_collate)
        #valset = BaseDataset(video_val_path, actions_dict, features_path, gt_path, pad_idx, n_class, n_query=args.n_query, args=args, query_dict=query_dict)

        valset = BaseDataset(video_val_path, actions_dict, features_path, depth_features_path, gt_path, pad_idx, n_class, n_query=args.n_query, args=args, query_dict=query_dict)
        val_loader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=1, collate_fn=valset.my_collate)

        
        train(args, model, train_loader, optimizer, scheduler, criterion,
                     model_save_path, pad_idx, device, val_loader, seed)


if __name__ == '__main__':
    # Seed fix
    seed = [1, 10, 13452]
    dataset_ops = 'baseline'
    main(seed[0], dataset_ops)
    main(seed[1], dataset_ops)
    main(seed[2], dataset_ops)
