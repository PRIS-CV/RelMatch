import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import os
import argparse, random, math
import copy, logging, sys, time, shutil, json
import torch.nn.functional as F
from aves_tree import *

def save_checkpoint(state, is_best, checkpoint_folder='exp',
                filename='checkpoint.pth.tar'):
    filename = os.path.join(checkpoint_folder, filename)
    best_model_filename = os.path.join(checkpoint_folder, 'model_best.pth.tar')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_model_filename)

def compute_correct(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        return res


def test(model, dataloaders, args, logger, name="Best", criterion=nn.CrossEntropyLoss(), relation_net=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_corrects_1 = 0
    test_corrects_5 = 0
    test_loss = 0
    
    for i,data in enumerate(dataloaders['test']):
        inputs, target = data
        inputs = inputs.to(device).float()
        target = target.to(device).long()

        ## upsample
        if args.input_size != inputs.shape[-1]:
            m = torch.nn.Upsample((args.input_size, args.input_size), mode='bilinear', align_corners=True)
            inputs = m(inputs)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, target)

            correct_1, correct_5 = compute_correct(outputs, target, topk=(1, 5))
            test_loss += loss.item()

        test_corrects_1 += correct_1.item()
        test_corrects_5 += correct_5.item()

    epoch_loss  = test_loss / i
    epoch_acc   = test_corrects_1 / len(dataloaders['test'].dataset)
    epoch_acc_5 = test_corrects_5 / len(dataloaders['test'].dataset)

    logger.info('{} Loss: {:.4f} Top1 Acc: {:.2f}% Top5 Acc: {:.2f}%'.format('test'+name, epoch_loss, epoch_acc*100, epoch_acc_5*100))
    
    return acc


            
def train_model(args, model, dataloaders, criterion, optimizer, 
    logger_name='train_logger', checkpoint_folder='exp',
    start_iter=0, best_acc=0.0, scheduler=None, relation_net=None):

    logger = logging.getLogger(logger_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())

    print_freq = args.print_freq

    iteration = start_iter
    running_loss = 0.0
    running_corrects_1 = 0
    
    ##################### code for tree ###################
    weight = torch.Tensor([0.2349, 0.3812, 0.1411, 0.1220, 0.1209]).cuda().float()
    CELoss_w = nn.CrossEntropyLoss(reduction='none', weight=weight)
    CELoss = nn.CrossEntropyLoss(reduction='none')
    trans_m = torch.Tensor([1,2,3,4,5]).unsqueeze(0).cuda()

    ####################
    ##### Training #####
    ####################
    for l_data, u_data in zip(dataloaders['l_train'], dataloaders['u_train']):
        iteration += 1 

        model.train()
        relation_net.train()
        
        l_input, target = l_data
        u_input, dummy_target = u_data

        l_input = l_input.to(device).float()
        u_input = u_input.to(device).float()
        dummy_target = dummy_target.to(device).long()
        target = target.to(device).long()

        ## upsample
        if args.input_size != l_input.shape[-1]:
            m = torch.nn.Upsample((args.input_size, args.input_size), mode='bilinear', align_corners=True)
            l_input = m(l_input)
            u_input = m(u_input)
        else:
            m = None
            
        with torch.set_grad_enabled(True):                
            outputs = model(l_input)
            f = outputs
            loss_c = CELoss(outputs, target)

            batch_size = f.size(0)
            f1 = outputs
            f2 = f1.flip(dims=[0])
            r = relation_net(f1, f2)
            t1 = target
            t2 = target.flip(dims=[0])
            t = get_tree_target(t1,t2)

            loss_r = (CELoss_w(r, t)).mean()

            cls_loss = loss_c + loss_r
            cls_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            correct_1, correct_5 = compute_correct(outputs, target, topk=(1, 5))  

            outputs = outputs.detach()

            acc_step = 1
            mu = 10
            ubs = batch_size * mu

            u_in = u_input[ubs * i: ubs * (i + 1)]
            feats, o = model(u_in, is_feat=True)
            f = o


            f1 = outputs.repeat(mu, 1)
            f2 = f1.flip(dims=[0])
            f3 = f
            t1 = target.repeat(mu)
            t2 = t1.flip(dims=[0])
            t = get_tree_target(t1,t2)

            for p in relation_net.parameters():
                p.requires_grad = False

            r1 = relation_net(f1, f3)
            r2 = relation_net(f2, f3)

            sc_r1 = torch.sum(F.softmax(r1, -1) * trans_m, 1).detach()
            sc_r2 = torch.sum(F.softmax(r2, -1) * trans_m, 1).detach()

            selection = sc_r1 < sc_r2
            ra = torch.cat((r1.unsqueeze(2), r2.unsqueeze(2)), 2)[torch.arange(batch_size * mu), :, selection.long()]

            mask2 = (torch.abs(sc_r1 - sc_r2) > 1).float()

            ssl_loss = (CELoss_w(ra, t) * mask2).mean() * 1
            ssl_loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            for p in relation_net.parameters():
                p.requires_grad = True

        running_loss += cls_loss.item() + ssl_loss.item()
        
        running_corrects_1 += correct_1.item()

        ## Print training loss/acc ##
        if (iteration+1) % print_freq==0:
            if args.alg != "supervised":

                else:
                    logger.info('{} | Iteration {:d}/{:d} | Loss {:f} | Top1 Acc {:.2f}%'.format( \
                        'train', iteration+1, len(dataloaders['l_train']), running_loss/print_freq, running_corrects_1*100/(print_freq*args.batch_size)))

            running_loss = 0.0
            running_corrects_1 = 0


        ####################
        #### Validation ####
        ####################
        if ((iteration+1) % args.val_freq) == 0 or (iteration+1) == args.num_iter:

            ## Print val loss/acc ##
            model.eval()
            val_loss = 0.0
            val_corrects_1 = 0
            val_corrects_5 = 0
            for i,data in enumerate(dataloaders['val']):
                inputs, target = data
                inputs = inputs.to(device).float()
                target = target.to(device).long()

                ## upsample
                if m is not None:
                    inputs = m(inputs)

                optimizer.zero_grad()
                with torch.set_grad_enabled(False):
                    outputs = model(inputs)
                    loss = criterion(outputs, target)

                    correct_1, correct_5 = compute_correct(outputs, target, topk=(1, 5))
                    val_loss += loss.item()

                val_corrects_1 += correct_1.item()
                val_corrects_5 += correct_5.item()

            num_val = len(dataloaders['val'].dataset)
            logger.info('{} | Iteration {:d}/{:d} | Loss {:f} | Top1 Acc {:.2f}% | Top5 Acc {:.2f}%'.format( 'Val', iteration+1, \
                args.num_iter, val_loss/i, val_corrects_1*100/num_val, val_corrects_5*100/num_val ))

            epoch_acc = val_corrects_1*100/num_val

            # deep copy the model with best val acc.
            is_best = epoch_acc > best_acc
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            val_acc_history.append(epoch_acc)

            save_checkpoint({
                'iteration': iteration + 1,
                'best_acc': best_acc,
                'model': args.model,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict' : optimizer.state_dict(),
                }, is_best, checkpoint_folder=checkpoint_folder)
        
        ## my setting
        if scheduler is None:
            ## Manually decrease lr if not using scheduler
            if (iteration+1)%args.lr_decay_iter == 0:
                optimizer.param_groups[0]["lr"] *= args.lr_decay_factor


    time_elapsed = time.time() - since
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logger.info('Best val Acc: {:.2f}%'.format(best_acc))


    ##############
    #### Test ####
    ##############.
    model.load_state_dict(best_model_wts)
    test(model,dataloaders,args,logger,"Best",relation_net=relation_net)

    return model, val_acc_history