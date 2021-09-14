# Copyright 2019 Christopher John Bayron
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file has been created by Christopher John Bayron based on "rnn_gan.py"
# by Olof Mogren. The referenced code is available in:
#
#     https://github.com/olofmogren/c-rnn-gan

import os, sys
import numpy as np
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch import optim

from c_rnn_gan import Generator, Discriminator
import utils
# import music_data_utils

MODEL_NAME = 'crnngan'
DATA_DIR = "../../data/processed_orig_data/"
GEN_DATA_DIR = "../../data/generated_data/crnngan/"

CKPT_DIR = 'models'
COMPOSER = 'sonata-ish'

G_FN = 'c_rnn_gan_g.pth'
D_FN = 'c_rnn_gan_d.pth'

G_LRN_RATE = 1e-3
D_LRN_RATE = 1e-3
MAX_GRAD_NORM = 5.

# following values are modified at runtime
CURR_SEQ_LEN = 24
MAX_SEQ_LEN = 200
BATCH_SIZE = 32

CLAMP_LB = 1e-40 # value to use to approximate zero (to prevent undefined results)
CLAMP_UB = 1.
PRINT_PERIOD = 50

class GLoss(nn.Module):
    ''' C-RNN-GAN generator loss
    '''
    def __init__(self):
        super(GLoss, self).__init__()

    def forward(self, logits_gen):
        # print('gl1: pre ', logits_gen.detach().mean())
        logits_gen2 = logits_gen
        logits_gen = torch.clamp(logits_gen, CLAMP_LB, CLAMP_UB)
        batch_loss = -torch.log(logits_gen)

        
        batch_loss2 = batch_loss
        # print('\ngl1: gen logit \ loss', logits_gen2.mean(), batch_loss2.mean())

        return torch.mean(batch_loss)


class GLoss2(nn.Module): 
    ''' C-RNN-GAN generator loss
    '''
    def __init__(self):
        super(GLoss2, self).__init__()

    def forward(self, d_logits_real, d_logits_gen):

        d_logits_real = torch.clamp(d_logits_real, CLAMP_LB, CLAMP_UB)
        d_logits_gen = torch.clamp(d_logits_gen, CLAMP_LB, CLAMP_UB)

        sq_err = (d_logits_real - d_logits_gen) ** 2
        mse1 = torch.sum(sq_err)

        d_logits_real = torch.mean(d_logits_real, dim=[2])
        d_logits_gen = torch.mean(d_logits_gen, dim=[2])
        sq_err_by_time = (d_logits_real - d_logits_gen) ** 2
        mse2 = torch.sum(sq_err)

        # d_logits_real2 = d_logits_real
        # d_logits_gen2 = d_logits_gen
        # mse1_2 = mse1
        # mse2_2 = mse2
        # print('\ng2:', d_logits_real2.detach().mean(), d_logits_gen2.detach().mean(), 
        #     mse1_2.detach().mean(), mse2_2.detach().mean()) 

        return mse1 + mse2


class DLoss(nn.Module):
    ''' C-RNN-GAN discriminator loss
    '''
    def __init__(self, label_smoothing=False):
        super(DLoss, self).__init__()
        self.label_smoothing = label_smoothing

    def forward(self, logits_real, logits_gen):
        ''' Discriminator loss

        logits_real: logits from D, when input is real
        logits_gen: logits from D, when input is from Generator

        loss = -(ylog(p) + (1-y)log(1-p))

        '''
        # print("\nd: real / gen logits ", logits_real.mean(), logits_gen.mean()) 

        logits_real = torch.clamp(logits_real, CLAMP_LB, CLAMP_UB)
        d_loss_real = -torch.log(logits_real)

        if self.label_smoothing:
            p_fake = torch.clamp((1 - logits_real), CLAMP_LB, CLAMP_UB)
            d_loss_fake = -torch.log(p_fake)
            d_loss_real = 0.9*d_loss_real + 0.1*d_loss_fake

        logits_gen = torch.clamp((1 - logits_gen), CLAMP_LB, CLAMP_UB)
        d_loss_gen = -torch.log(logits_gen)

        batch_loss = d_loss_real + d_loss_gen

        # print("d: real / gen loss ", d_loss_real.mean(), d_loss_gen.mean()) 


        return torch.mean(batch_loss)



def run_training(model, optimizer, criterion, dataloader, freeze_g=False, freeze_d=False):
    ''' Run single training epoch
    '''
    
    num_feats = dataloader.get_num_seq_features()
    dataloader.rewind(part='train')
    batch_meta, batch_song = dataloader.get_batch(BATCH_SIZE, CURR_SEQ_LEN, part='train')

    model['g'].train()
    model['d'].train()

    loss = {}
    g_loss_total = 0.0
    d_loss_total = 0.0
    num_corrects = 0
    num_sample = 0

    while batch_song is not None:

        real_batch_sz = batch_song.shape[0]

        # get initial states
        # each batch is independent i.e. not a continuation of previous batch
        # so we reset states for each batch
        # POSSIBLE IMPROVEMENT: next batch is continuation of previous batch
        g_states = model['g'].init_hidden(real_batch_sz)
        d_state = model['d'].init_hidden(real_batch_sz)
        #===========================================================================================
        #### GENERATOR ####
        if not freeze_g:
            optimizer['g'].zero_grad()
        # prepare inputs
        z = torch.empty([real_batch_sz, CURR_SEQ_LEN, num_feats]).uniform_() # random vector
        batch_song = torch.Tensor(batch_song)

        # feed inputs to generator
        g_feats, _ = model['g'](z, g_states)

        # calculate loss, backprop, and update weights of G
        if isinstance(criterion['g'], GLoss):
            d_logits_gen, _, _ = model['d'](g_feats, d_state)
            loss['g'] = criterion['g'](d_logits_gen)
        else: # feature matching
            # feed real and generated input to discriminator
            _, d_feats_real, _ = model['d'](batch_song, d_state)
            _, d_feats_gen, _ = model['d'](g_feats, d_state)

            loss['g'] = criterion['g'](d_feats_real, d_feats_gen)
            # loss['g'] = criterion['g'](batch_song, g_feats)


        if not freeze_g:
            loss['g'].backward()
            nn.utils.clip_grad_norm_(model['g'].parameters(), max_norm=MAX_GRAD_NORM)
            optimizer['g'].step()

        #===========================================================================================
        #### DISCRIMINATOR ####
        if not freeze_d:
            optimizer['d'].zero_grad()
        # feed real and generated input to discriminator
        d_logits_real, _, _ = model['d'](batch_song, d_state)
        # need to detach from operation history to prevent backpropagating to generator
        d_logits_gen, _, _ = model['d'](g_feats.detach(), d_state)

        # print("loop d: mean", d_logits_real.mean(), d_logits_gen.mean())
        # print("loop d: real min/max ", d_logits_real.min(), d_logits_real.max())
        # print("loop d: gen  max/max ", d_logits_gen.min(), d_logits_gen.max())

        # calculate loss, backprop, and update weights of D
        loss['d'] = criterion['d'](d_logits_real, d_logits_gen)
        if not freeze_d:
            loss['d'].backward()
            nn.utils.clip_grad_norm_(model['d'].parameters(), max_norm=MAX_GRAD_NORM)
            optimizer['d'].step()

        

        g_loss_total += loss['g'].item()
        d_loss_total += loss['d'].item()

        batch_correct = (d_logits_real > 0.5).sum().item() + (d_logits_gen < 0.5).sum().item()
        num_corrects += batch_correct

        # print("loop acc: ", (0.5 * batch_correct / real_batch_sz), batch_correct, real_batch_sz)

        num_sample += real_batch_sz

        # fetch next batch
        batch_meta, batch_song = dataloader.get_batch(BATCH_SIZE, CURR_SEQ_LEN, part='train')

    g_loss_avg, d_loss_avg = 0.0, 0.0
    d_acc = 0.0
    if num_sample > 0:
        g_loss_avg = g_loss_total / num_sample
        d_loss_avg = d_loss_total / num_sample
        d_acc = 100 * num_corrects / (2 * num_sample) # 2 because (real + generated)

    return model, g_loss_avg, d_loss_avg, d_acc


def run_validation(model, criterion, dataloader):
    ''' Run single validation epoch
    '''
    num_feats = dataloader.get_num_seq_features()
    dataloader.rewind(part='validation')
    batch_meta, batch_song = dataloader.get_batch(BATCH_SIZE, CURR_SEQ_LEN, part='validation')

    model['g'].eval()
    model['d'].eval()

    g_loss_total = 0.0
    d_loss_total = 0.0
    num_corrects = 0
    num_sample = 0

    while batch_song is not None:

        real_batch_sz = batch_song.shape[0]

        # initial states
        g_states = model['g'].init_hidden(real_batch_sz)
        d_state = model['d'].init_hidden(real_batch_sz)

        #### GENERATOR ####
        # prepare inputs
        z = torch.empty([real_batch_sz, CURR_SEQ_LEN, num_feats]).uniform_() # random vector
        batch_song = torch.Tensor(batch_song)

        # feed inputs to generator
        g_feats, _ = model['g'](z, g_states)
        # feed real and generated input to discriminator
        d_logits_real, d_feats_real, _ = model['d'](batch_song, d_state)
        d_logits_gen, d_feats_gen, _ = model['d'](g_feats, d_state)
        # calculate loss
        if isinstance(criterion['g'], GLoss):
            g_loss = criterion['g'](d_logits_gen)
        else: # feature matching
            g_loss = criterion['g'](d_feats_real, d_feats_gen)

        d_loss = criterion['d'](d_logits_real, d_logits_gen)

        g_loss_total += g_loss.item()
        d_loss_total += d_loss.item()
        num_corrects += (d_logits_real > 0.5).sum().item() + (d_logits_gen < 0.5).sum().item()
        num_sample += real_batch_sz

        # fetch next batch
        batch_meta, batch_song = dataloader.get_batch(BATCH_SIZE, CURR_SEQ_LEN, part='validation')

    g_loss_avg, d_loss_avg = 0.0, 0.0
    d_acc = 0.0
    if num_sample > 0:
        g_loss_avg = g_loss_total / num_sample
        d_loss_avg = d_loss_total / num_sample
        d_acc = 100 * num_corrects / (2 * num_sample) # 2 because (real + generated)

    return g_loss_avg, d_loss_avg, d_acc


def run_epoch(model, optimizer, criterion, dataloader, ep, num_ep,
              freeze_g=False, freeze_d=False, pretraining=False):
    ''' Run a single epoch
    '''
    model, trn_g_loss, trn_d_loss, trn_acc = \
        run_training(model, optimizer, criterion, dataloader, freeze_g=freeze_g, freeze_d=freeze_d)

    val_g_loss, val_d_loss, val_acc = run_validation(model, criterion, dataloader)

    if ep % PRINT_PERIOD == PRINT_PERIOD - 1:
        print("-"*80)
        if pretraining:
            print("Pretraining Epoch %d/%d " % (ep+1, num_ep), "[Freeze G: ", freeze_g, ", Freeze D: ", freeze_d, "]")
        else:
            print("Epoch %d/%d " % (ep+1, num_ep), "[Freeze G: ", freeze_g, ", Freeze D: ", freeze_d, "]")

        print("\t[Training] G_loss: %0.8f, D_loss: %0.8f, D_acc: %0.2f\n"
            "\t[Validation] G_loss: %0.8f, D_loss: %0.8f, D_acc: %0.2f" %
            (trn_g_loss, trn_d_loss, trn_acc,
            val_g_loss, val_d_loss, val_acc))

    return model, trn_acc


def generate_samples(model, num_samples, num_feats): 
    g_states = model['g'].init_hidden(num_samples)

    z = torch.empty([num_samples, CURR_SEQ_LEN, num_feats]).uniform_() # random vector

    if torch.cuda.is_available():
        z = z.cuda()
        model['g'].cuda()

    model['g'].eval()
    g_feats, _ = model['g'](z, g_states)    
    # g_feats = g_feats.squeeze().cpu()
    song_data = g_feats.detach().numpy()
    return song_data



def main(args):
    ''' Training sequence
    '''
    print(args) ; # sys.exit()

    dataloader = utils.DataLoader(args.dataset, args.training_thresh, DATA_DIR, BATCH_SIZE,
            args.valid_perc, test_perc = 0, do_shuffle = True) 

    num_feats = dataloader.get_num_seq_features()
    seq_len = dataloader.get_seq_len()
    # print(num_feats, seq_len); sys.exit()

    del dataloader

    global CURR_SEQ_LEN


    # First checking if GPU is available
    train_on_gpu = torch.cuda.is_available()
    if train_on_gpu:
        print('Training on GPU.')
    else:
        print('No GPU available, training on CPU.')     

    model = {
        'g': Generator(num_feats, use_cuda=train_on_gpu),
        'd': Discriminator(num_feats, use_cuda=train_on_gpu)
    }

    if args.use_sgd:
        optimizer = {
            'g': optim.SGD(model['g'].parameters(), lr=args.g_lrn_rate, momentum=0.9),
            'd': optim.SGD(model['d'].parameters(), lr=args.d_lrn_rate, momentum=0.9),
        }
    else:
        optimizer = {
            'g': optim.Adam(model['g'].parameters(), args.g_lrn_rate),
            'd': optim.Adam(model['d'].parameters(), args.d_lrn_rate)
        }

    criterion = {
        'g': nn.MSELoss(reduction='sum') if args.feature_matching else GLoss(),
        # 'g': GLoss2() if args.feature_matching else GLoss(),
        'd': DLoss(args.label_smoothing)
    }

    if args.load_g:
        ckpt = torch.load(os.path.join(CKPT_DIR, G_FN))
        model['g'].load_state_dict(ckpt)
        print("Continue training of %s" % os.path.join(CKPT_DIR, G_FN))

    if args.load_d:
        ckpt = torch.load(os.path.join(CKPT_DIR, D_FN))
        model['d'].load_state_dict(ckpt)
        print("Continue training of %s" % os.path.join(CKPT_DIR, D_FN))

    if train_on_gpu:
        model['g'].cuda()
        model['d'].cuda()


    ####################################################################################################################
    
    start = 2
    end = 12
    step = 2

    lens = [i * step for i in np.arange(start, end+1)]
    
    for used_len in lens: 

        CURR_SEQ_LEN = used_len

        print('#'*80)
        print("Changing seq_len to ", used_len)

        dataloader = utils.DataLoader2(args.dataset, args.training_thresh, DATA_DIR, BATCH_SIZE,
            args.valid_perc, test_perc = 0, used_len = used_len, do_shuffle = True) 

        num_feats = dataloader.get_num_seq_features()
        seq_len = dataloader.get_seq_len()        

        if not args.no_pretraining:
            for ep in range(args.g_pretraining_epochs):
                model, trn_acc = run_epoch(model, optimizer, criterion, dataloader,
                                ep, args.g_pretraining_epochs, freeze_d=True, pretraining=True)
            
            for ep in range(args.d_pretraining_epochs):
                model, trn_acc = run_epoch(model, optimizer, criterion, dataloader,
                                ep, args.d_pretraining_epochs, freeze_g=True, pretraining=True)

        
        # freeze_d = False
        # for ep in range(args.num_epochs):

        #     model, trn_acc = run_epoch(model, optimizer, criterion, dataloader, ep, args.num_epochs, freeze_d=freeze_d)
        #     if args.conditional_freezing:
        #         # conditional freezing
        #         freeze_d = False
        #         if trn_acc >= 90.0:
        #             freeze_d = True
        
        ep = 0
        while ep < args.num_epochs * 2: # 2 because counting generator and discri separately in iter count
            if trn_acc >= 60.: 
                # print("-----starting generator training --------", trn_acc, ep)
                while trn_acc >= 60. and ep < args.num_epochs * 2:
                    model, trn_acc = run_epoch(model, optimizer, criterion, dataloader,
                                    ep//2, args.num_epochs, freeze_d=True)
                    ep += 1
                # print("-----ended generator training --------", trn_acc)
            
            if trn_acc < 60.:
                # print("=====started discriminator training =======",  trn_acc, ep)
                while trn_acc < 80. and ep < args.num_epochs * 2:
                    model, trn_acc = run_epoch(model, optimizer, criterion, dataloader,
                                    ep//2, args.num_epochs, freeze_g=True)
                    ep += 1
                # print("=====ended discriminator training =======",  trn_acc)

        #     # sys.exit()
        
    if not args.no_save_g:
        torch.save(model['g'].state_dict(), os.path.join(CKPT_DIR, G_FN))
        print("Saved generator: %s" % os.path.join(CKPT_DIR, G_FN))

    if not args.no_save_d:
        torch.save(model['d'].state_dict(), os.path.join(CKPT_DIR, D_FN))
        print("Saved discriminator: %s" % os.path.join(CKPT_DIR, D_FN))

    samples = generate_samples(model, dataloader.get_orig_num_samples(), num_feats)
    samples = dataloader.get_rescaled_data(samples)
    samples_fpath = f'{MODEL_NAME}_gen_samples_{args.dataset}_perc_{args.training_thresh}.npz'        
    np.savez_compressed(os.path.join( GEN_DATA_DIR, samples_fpath), data=samples)


if __name__ == "__main__":

    ARG_PARSER = ArgumentParser()

    ARG_PARSER.add_argument('--dataset', default='sine')
    ARG_PARSER.add_argument('--training_thresh', default=20, type=int)

    ARG_PARSER.add_argument('--valid_perc', default=10, type=int)

    ARG_PARSER.add_argument('--load_g', action='store_true')
    ARG_PARSER.add_argument('--load_d', action='store_true')
    ARG_PARSER.add_argument('--no_save_g', action='store_true')
    ARG_PARSER.add_argument('--no_save_d', action='store_true')

    ARG_PARSER.add_argument('--seq_len', default=24, type=int)
    ARG_PARSER.add_argument('--batch_size', default=32, type=int)
    ARG_PARSER.add_argument('--g_lrn_rate', default=1e-3, type=float)
    ARG_PARSER.add_argument('--d_lrn_rate', default=1e-1, type=float)

    ARG_PARSER.add_argument('--no_pretraining', action='store_true')
    ARG_PARSER.add_argument('--g_pretraining_epochs', default=50, type=int)
    ARG_PARSER.add_argument('--d_pretraining_epochs', default=20, type=int)
    # ARG_PARSER.add_argument('--freeze_d_every', default=5, type=int)
    ARG_PARSER.add_argument('--use_sgd', action='store_true')

    ARG_PARSER.add_argument('--conditional_freezing', action='store_true')    
    ARG_PARSER.add_argument('--label_smoothing', action='store_true')

    ARG_PARSER.add_argument('--feature_matching', action='store_true')

    ARG_PARSER.add_argument('--num_epochs', default=200, type=int)

    ARGS = ARG_PARSER.parse_args()
    MAX_SEQ_LEN = ARGS.seq_len
    BATCH_SIZE = ARGS.batch_size

    main(ARGS)
