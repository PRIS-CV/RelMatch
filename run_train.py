import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

import os, argparse, random, math
import copy, logging, sys, time, shutil, json
from collections import Counter, OrderedDict

from lib import wrn, transform
from lib.initialize import initialize_model
from training import *
from lib.datasets.iNatDataset import iNatDataset
from lib.model_lib import Relation_Net

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

dset_root = {}
dset_root['semi_fungi'] = './data/semi_fungi'
dset_root['semi_aves'] = './data/semi_aves'


class RandomSampler(torch.utils.data.Sampler):
    """ sampling without replacement """
    def __init__(self, num_data, num_sample):
        iterations = num_sample // num_data + 1
        self.indices = torch.cat([torch.randperm(num_data) for _ in range(iterations)]).tolist()[:num_sample]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def initializeLogging(log_filename, logger_name):
    log = logging.getLogger(logger_name)
    log.setLevel(logging.DEBUG)
    log.addHandler(logging.StreamHandler(sys.stdout))
    log.addHandler(logging.FileHandler(log_filename, mode='a'))
    return log

def seed_torch(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def main(args):
#     seed_torch(seed)
    log_dir = os.path.join(args.exp_prefix, args.exp_dir, 'log')
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    else:
        os.makedirs(log_dir)

    json.dump(dict(sorted(vars(args).items())), open(os.path.join(args.exp_prefix, args.exp_dir, 'configs.json'),'w'))

    checkpoint_folder = os.path.join(args.exp_prefix, args.exp_dir, 'checkpoints')
    if not os.path.isdir(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    logger_name = 'train_logger'
    logger = initializeLogging(os.path.join(args.exp_prefix, args.exp_dir, 'train_history.txt'), logger_name)

    # ==================  Craete data loader ==================================
    data_transforms = {
        'train': transforms.Compose([
#             transforms.Resize(args.input_size), 
            transforms.RandomResizedCrop(args.input_size),
            # transforms.ColorJitter(Brightness=0.4, Contrast=0.4, Color=0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(args.input_size), 
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    data_transforms['l_train'] = data_transforms['train']
    data_transforms['u_train'] = data_transforms['train']
    data_transforms['val'] = data_transforms['test']

    root_path = dset_root[args.task]

    if args.trainval:
        ## use l_train + val for labeled training data
        l_train = 'l_train_val'
    else:
        l_train = 'l_train'

    if args.unlabel == 'in':
        u_train = 'u_train_in'
    elif args.unlabel == 'inout':
        u_train = 'u_train'

    ## set val to test when using l_train + val for training
    if args.trainval:
        split_fname = ['test', 'test']
    else:
        split_fname = ['val', 'test']

    image_datasets = {split: iNatDataset(root_path, split_fname[i], args.task,
        transform=data_transforms[split]) \
        for i,split in enumerate(['val', 'test'])}
    image_datasets['u_train'] = iNatDataset(root_path, u_train, args.task,
        transform=data_transforms['u_train'])
    image_datasets['l_train'] = iNatDataset(root_path, l_train, args.task,
        transform=data_transforms['train'])

    print("labeled data : {}, unlabeled data : {}".format(len(image_datasets['l_train']), len(image_datasets['u_train'])))
    print("validation data : {}, test data : {}".format(len(image_datasets['val']), len(image_datasets['test'])))

    num_classes = image_datasets['l_train'].get_num_classes() 
    
    print("#classes : {}".format(num_classes))

    dataloaders_dict = {}
    dataloaders_dict['l_train'] = DataLoader(image_datasets['l_train'],
                    batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True,
                    sampler=RandomSampler(len(image_datasets['l_train']), args.num_iter * args.batch_size))

    mu = 10
    dataloaders_dict['u_train'] = DataLoader(image_datasets['u_train'],
                    batch_size=args.batch_size * mu, num_workers=args.num_workers, drop_last=True,
                    sampler=RandomSampler(len(image_datasets['u_train']), args.num_iter * args.batch_size * mu))
    dataloaders_dict['val'] = DataLoader(image_datasets['val'],
                    batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    dataloaders_dict['test'] = DataLoader(image_datasets['test'],
                    batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False,
                    sampler=RandomSampler(len(image_datasets['test']), len(image_datasets['test'])))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_ft = initialize_model(args.model, num_classes, feature_extract=False, 
                    use_pretrained=True, logger=logger)
    relation_net = Relation_Net(feature_size=256)

    optimizer = optim.SGD([{'params': model_ft.parameters(), 'lr': args.lr},
                           {'params': relation_net.parameters(), 'lr': args.lr}, 
                          ], momentum=0.9, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_iter)

    start_iter = 0
    best_acc = 0.0
    model_ft = torch.nn.DataParallel(model_ft)
    relation_net = torch.nn.DataParallel(relation_net)
    model_ft.to(device)
    relation_net.to(device)

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    print("parameters : ", args)
    model_ft, val_acc_history = train_model(args, model_ft, dataloaders_dict, criterion, optimizer,
            logger_name=logger_name, checkpoint_folder=checkpoint_folder,
            start_iter=start_iter, best_acc=best_acc, ssl_obj=None, scheduler=scheduler, relation_net=relation_net)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='semi_aves', type=str, 
            help='the name of the dataset')
    parser.add_argument('--model', default='resnet50', type=str,
            help='resnet50|resnet101')
    parser.add_argument('--batch_size', default=32, type=int,
            help='size of mini-batch')
    parser.add_argument('--num_iter', default=200, type=int,
            help='number of iterations')
    parser.add_argument('--exp_prefix', default='results', type=str,
            help='path to the chekcpoint folder for the experiment')
    parser.add_argument('--exp_dir', default='exp', type=str,
            help='path to the chekcpoint folder for the experiment')
    parser.add_argument('--load_dir', default='', type=str,
            help='load pretrained model from')
    parser.add_argument('--input_size', default=224, type=int, 
            help='input image size')
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--val_freq', default=200, type=int,
            help='do val every x iter')
    parser.add_argument('--print_freq', default=100, type=int,
            help='show train loss/acc every x iter')
    parser.add_argument("--wd", default=1e-4, type=float, 
            help="weight decay")
    parser.add_argument('--trainval', action='store_true', 
            help='use {train+val,test,test} for {train,val,test}')
    parser.add_argument("--lr", default=1e-3, type=float, 
            help="learning rate")
    parser.add_argument('--unlabel', default='in', type=str, 
            choices=['in','inout'], help='U_in or U_in + U_out')

    args = parser.parse_args()

    
    main(args)



