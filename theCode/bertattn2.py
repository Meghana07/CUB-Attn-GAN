import os
os.chdir('/content/AttnGAN/code/')

import os.path as osp
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import numpy as np
import numpy.random as random
import pandas as pd
import pickle
from easydict import EasyDict as edict

from PIL import Image, ImageDraw, ImageFont
from copy import deepcopy
import skimage.transform

import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from torchvision import models
import torchvision.transforms as transforms

import torch.utils.model_zoo as model_zoo
import torch.utils.data as data

from nltk.tokenize import RegexpTokenizer
from collections import defaultdict


__C = edict()
cfg = __C
__C.DATASET_NAME = 'birds'
__C.CONFIG_NAME = 'DAMSM'
__C.DATA_DIR = '../data/birds'
__C.GPU_ID = 0
__C.CUDA = True
__C.WORKERS = 1
__C.B_VALIDATION = False

__C.TREE = edict()
__C.TREE.BRANCH_NUM = 1
__C.TREE.BASE_SIZE = 299

# Training options
__C.TRAIN = edict()
__C.TRAIN.BATCH_SIZE = 48
__C.TRAIN.MAX_EPOCH = 600
__C.TRAIN.SNAPSHOT_INTERVAL = 50
__C.TRAIN.DISCRIMINATOR_LR = 0.0002
__C.TRAIN.GENERATOR_LR = 0.0002
__C.TRAIN.ENCODER_LR = 0.002
__C.TRAIN.RNN_GRAD_CLIP = 0.25
__C.TRAIN.FLAG = True
__C.TRAIN.NET_E = ''
__C.TRAIN.NET_G = ''
__C.TRAIN.B_NET_D = True
__C.TRAIN.SMOOTH = edict()
__C.TRAIN.SMOOTH.GAMMA1 = 4.0
__C.TRAIN.SMOOTH.GAMMA3 = 10.0
__C.TRAIN.SMOOTH.GAMMA2 = 5.0
__C.TRAIN.SMOOTH.LAMBDA = 1.0

# Modal options
__C.GAN = edict()
__C.GAN.DF_DIM = 64
__C.GAN.GF_DIM = 128
__C.GAN.Z_DIM = 100
__C.GAN.CONDITION_DIM = 100
__C.GAN.R_NUM = 2
__C.GAN.B_ATTENTION = True
__C.GAN.B_DCGAN = False

__C.TEXT = edict()
__C.TEXT.CAPTIONS_PER_IMAGE = 10
__C.TEXT.EMBEDDING_DIM = 256
__C.TEXT.WORDS_NUM = 18



def get_imgs(img_path, imsize, bbox=None, transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])

    if transform is not None:
        img = transform(img)

    ret = []
    if cfg.GAN.B_DCGAN:
        ret = [normalize(img)]
    else:
        for i in range(cfg.TREE.BRANCH_NUM):
            # print(imsize[i])
            if i < (cfg.TREE.BRANCH_NUM - 1):
                re_img = transforms.Scale(imsize[i])(img)
            else:
                re_img = img
            ret.append(normalize(re_img))

    return ret

def prepare_data(data):
    imgs, captions, captions_lens, class_ids, keys, input_ids, segments_ids = data

    # sort data by the length in a decreasing order !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!MARKER!!!!!!!!!!!!!!!!!!!!!!!!
    sorted_cap_lens, sorted_cap_indices = torch.sort(captions_lens, 0, True)

    real_imgs = []
    for i in range(len(imgs)):
        imgs[i] = imgs[i][sorted_cap_indices]
        if cfg.CUDA:
            real_imgs.append(Variable(imgs[i]).cuda())
        else:
            real_imgs.append(Variable(imgs[i]))

    captions = captions[sorted_cap_indices].squeeze()
    class_ids = class_ids[sorted_cap_indices].numpy()
    # sent_indices = sent_indices[sorted_cap_indices]
    keys = [keys[i] for i in sorted_cap_indices.numpy()]
    # print('keys', type(keys), keys[-1])  # list
    if cfg.CUDA:
        captions = Variable(captions).cuda()
        sorted_cap_lens = Variable(sorted_cap_lens).cuda()
    else:
        captions = Variable(captions)
        sorted_cap_lens = Variable(sorted_cap_lens)

    input_ids = input_ids.cuda()
    segments_ids = segments_ids.cuda()

    return [real_imgs, captions, sorted_cap_lens,
            class_ids, keys, input_ids, segments_ids]

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def build_super_images(real_imgs, captions, ixtoword,
                        attn_maps, att_sze, lr_imgs=None,
                        batch_size=cfg.TRAIN.BATCH_SIZE,
                        max_word_num=cfg.TEXT.WORDS_NUM):
    
    
    COLOR_DIC = {0:[128,64,128],  1:[244, 35,232],
                2:[70, 70, 70],  3:[102,102,156],
                4:[190,153,153], 5:[153,153,153],
                6:[250,170, 30], 7:[220, 220, 0],
                8:[107,142, 35], 9:[152,251,152],
                10:[70,130,180], 11:[220,20, 60],
                12:[255, 0, 0],  13:[0, 0, 142],
                14:[119,11, 32], 15:[0, 60,100],
                16:[0, 80, 100], 17:[0, 0, 230],
                18:[0,  0, 70],  19:[0, 0,  0]}
    FONT_MAX = 50

    
    build_super_images_start_time = time.time()
    nvis = 8
    real_imgs = real_imgs[:nvis]
    if lr_imgs is not None:
        lr_imgs = lr_imgs[:nvis]
    if att_sze == 17:
        vis_size = att_sze * 16
    else:
        vis_size = real_imgs.size(2)

    text_convas = \
        np.ones([batch_size * FONT_MAX,
                 (max_word_num + 2) * (vis_size + 2), 3],
                dtype=np.uint8)


    # print("keyword |||||||||||||||||||||||||||||||")
    # print("max_word_num : " , max_word_num)
    # print("keyword |||||||||||||||||||||||||||||||")
    for i in range(max_word_num):
        istart = (i + 2) * (vis_size + 2)
        iend = (i + 3) * (vis_size + 2)
        text_convas[:, istart:iend, :] = COLOR_DIC[i]


    real_imgs = \
        nn.Upsample(size=(vis_size, vis_size), mode='bilinear')(real_imgs)
    # [-1, 1] --> [0, 1]
    real_imgs.add_(1).div_(2).mul_(255)
    real_imgs = real_imgs.data.numpy()
    # b x c x h x w --> b x h x w x c
    real_imgs = np.transpose(real_imgs, (0, 2, 3, 1))
    pad_sze = real_imgs.shape
    middle_pad = np.zeros([pad_sze[2], 2, 3])
    post_pad = np.zeros([pad_sze[1], pad_sze[2], 3])
    if lr_imgs is not None:
        lr_imgs = \
            nn.Upsample(size=(vis_size, vis_size), mode='bilinear')(lr_imgs)
        # [-1, 1] --> [0, 1]
        lr_imgs.add_(1).div_(2).mul_(255)
        lr_imgs = lr_imgs.data.numpy()
        # b x c x h x w --> b x h x w x c
        lr_imgs = np.transpose(lr_imgs, (0, 2, 3, 1))

    # batch x seq_len x 17 x 17 --> batch x 1 x 17 x 17
    seq_len = max_word_num
    img_set = []
    num = nvis  # len(attn_maps)

    text_map, sentences = \
        drawCaption(text_convas, captions, ixtoword, vis_size)
    text_map = np.asarray(text_map).astype(np.uint8)

    bUpdate = 1
    for i in range(num):
        #print ("loop " , i ," of " , num == 8)
        attn = attn_maps[i].cpu().view(1, -1, att_sze, att_sze)
        # --> 1 x 1 x 17 x 17
        attn_max = attn.max(dim=1, keepdim=True)
        attn = torch.cat([attn_max[0], attn], 1)
        #
        attn = attn.view(-1, 1, att_sze, att_sze)
        attn = attn.repeat(1, 3, 1, 1).data.numpy()
        # n x c x h x w --> n x h x w x c
        attn = np.transpose(attn, (0, 2, 3, 1))
        num_attn = attn.shape[0]
        #
        img = real_imgs[i]
        if lr_imgs is None:
            lrI = img
        else:
            lrI = lr_imgs[i]
        
        row = [lrI, middle_pad]
        #print("rowwwwwwwwwwwwwwwww : ", row)
        row_merge = [img, middle_pad]
        row_beforeNorm = []
        minVglobal, maxVglobal = 1, 0
        for j in range(num_attn):
            #print ("looop " , j , " of " , seq_len+1)
            one_map = attn[j]
            #print("First one map : " , one_map.shape)
            #print("attn.shape : " , attn.shape)

            
            # print("if (vis_size // att_sze) > 1: " ,  (vis_size // att_sze) > 1)
            # print("vis_size : " , vis_size)
            # print("att_sze : " , att_sze)
            # print("vis_size//att_sze : " , vis_size//att_sze)
            
            if (vis_size // att_sze) > 1:
                one_map = \
                    skimage.transform.pyramid_expand(one_map, sigma=20,
                                                     upscale=vis_size // att_sze)
            #    print("one_map in if : " , one_map.shape)

            
            row_beforeNorm.append(one_map)
            #print("row_beforeNorm.append(one_map)" ,len(row_beforeNorm))
            minV = one_map.min()
            maxV = one_map.max()
            if minVglobal > minV:
                minVglobal = minV
            if maxVglobal < maxV:
                maxVglobal = maxV
            #print("seq_len : " , seq_len)
        for j in range(seq_len + 1):
            #print ("loooop " , j , " of " , seq_len+1)
            
            if j < num_attn:
                one_map = row_beforeNorm[j]
                one_map = (one_map - minVglobal) / (maxVglobal - minVglobal)
                one_map *= 255
                #
                # print ("PIL_im = " , Image.fromarray(np.uint8(img)))
                # print ("PIL_att = " , Image.fromarray(np.uint8(one_map[:,:,:3])))
                # print ("img.size( :" , img.shape)
                # print ("one_map.size( :" , one_map.shape)
                PIL_im = Image.fromarray(np.uint8(img))
                PIL_att = Image.fromarray(np.uint8(one_map[:,:,:3]))
                merged = \
                    Image.new('RGBA', (vis_size, vis_size), (0, 0, 0, 0))
                #print ("merged : " , merged.size)
                mask = Image.new('L', (vis_size, vis_size), (210))
                #print (" mask  : " , mask.size)
                merged.paste(PIL_im, (0, 0))
                #print (" merged.paste(PIL_im)  : " , merged.size )
                ############################################################
                merged.paste(PIL_att, (0, 0), mask)
                #print (" merged.paste(PIL_att)  : " ,  merged.size)#########################
                merged = np.array(merged)[:, :, :3]
                #print ("  np.array(merged)[:::3] : " , merged.size )#########################
                ############################################################
            else:
                #print (" IN THE ELSE post_pad : " , post_pad.shape)
                one_map = post_pad
                #print (" one_map  : " , one_map.shape )
                merged = post_pad
                #print ("  OUTTING THE ELSE : " , merged.shape )
            
            #print ("  row : " , len(row))
            row.append(one_map[:,:,:3])
            #print ("  row.appedn(one_map) : " , len(row))
            row.append(middle_pad)
            #print ("  row.append(middle_pad) : " , len(row))
            #
            #print ("  row_merge : " , len(row_merge))
            row_merge.append(merged)
            #print ("  row_merge.append(mereged) : " , len(row_merge) )
            row_merge.append(middle_pad)
            #print ("  row_merge.append(middle_pad) : " , len(row_merge) )
        ####################################################################
        # print("row.shape : ", len(row))
        # for i in range(len(row)):
        #   print('arr', i,   
        #         " => dim0:", len(row[i]),
        #         " || dim1:", len(row[i][0]),
        #         " || dim2:", len(row[i][0][0]))
        # #print(row)
        # print("row[0].shape : ", len(row[0]))
        # #print(row[0])
        # print("row[0][0].shape : ", len(row[0][0]))
        # #print(row[0][0])
        # print("row[0][0][0].shape : ", len(row[0][0][0]))
        # #print(row[0][0][0])

        # print("row[1].shape : ", len(row[1]))
        # #print(row[1])
        # print("row[1][0].shape : ", len(row[1][0]))
        # #print(row[1][0])
        # print("row[1][0][0].shape : ", len(row[1][0][0]))
        # #print(row[1][0][0])

        # print("row[2].shape : ", len(row[2]))
        # #print(row[2])
        # print("row[2][0].shape : ", len(row[2][0]))
        # #print(row[2][0])
        # print("row[2][0][0].shape : ", len(row[2][0][0]))
        # #print(row[2][0][0])

        # print("row[3].shape : ", len(row[3]))
        # #print(row[2])
        # print("row[3][0].shape : ", len(row[3][0]))
        # #print(row[2][0])
        # print("row[3][0][0].shape : ", len(row[3][0][0]))
        # #print(row[2][0][0])

        # print("row[4].shape : ", len(row[4]))
        # #print(row[2])
        # print("row[4][0].shape : ", len(row[4][0]))
        # #print(row[2][0])
        # print("row[4][0][0].shape : ", len(row[4][0][0]))
        #print(row[2][0][0])

        

        
        
        row = np.concatenate(row, 1)
        #print (" row.conatent(1)  : " ,  len(row))########################################
        row_merge = np.concatenate(row_merge, 1)
        #print ("   : " , )############################
        ####################################################################
        txt = text_map[i * FONT_MAX: (i + 1) * FONT_MAX]
        if txt.shape[1] != row.shape[1]:
            print('txt', txt.shape, 'row', row.shape)
            bUpdate = 0
            break
        #####################################################################
        row = np.concatenate([txt, row, row_merge], 0)#######################
        img_set.append(row)##################################################
        #####################################################################
    
    # print("keyword |||||||||||||||||||||||||||||||")
    # print("bUpdate : " , bUpdate)
    # print("keyword |||||||||||||||||||||||||||||||")
    if bUpdate:
        img_set = np.concatenate(img_set, 0)
        img_set = img_set.astype(np.uint8)
        print("keyTime |||||||||||||||||||||||||||||||")
        print("build_super_images_time : " , time.time() - build_super_images_start_time)
        print("KeyTime |||||||||||||||||||||||||||||||")
        return img_set, sentences
    else:
        print("keyTime |||||||||||||||||||||||||||||||")
        print("build_super_images_start_time : " , time.time() - build_super_images_start_time)
        print("KeyTime |||||||||||||||||||||||||||||||")
        return None

def conv1x1(in_planes, out_planes, bias=False):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=bias)


class TextDataset(data.Dataset):
    def __init__(self, data_dir, split='train',
                    base_size=64,
                    transform=None, target_transform=None, input_ids=None, segments_ids=None, sentences=None):
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.target_transform = target_transform
        self.embeddings_num = cfg.TEXT.CAPTIONS_PER_IMAGE

        self.imsize = []# [299]
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2
        print("self.imsize", self.imsize)

        self.data = []
        self.data_dir = data_dir
        if data_dir.find('birds') != -1:
            self.bbox = self.load_bbox() # 11788 long dictionry with key as image name and value is 4 ints list bounding box
        else:
            self.bbox = None
        split_dir = os.path.join(data_dir, split)

        self.filenames, self.captions, self.ixtoword, self.wordtoix, self.n_words = self.load_text_data(data_dir, split)
        #filenames: List of 8855 text items of image names
        #captions: List of 8855 varible lengths captions -in range 9-18 -
        #ixtoword: dictionry  of 5450 index [key] to word [value] pairs
        #wordtoix: dictionry  of 5450 word [key] to index [value] pairs
        #n_words: 5450

        self.class_id = self.load_class_id(split_dir, len(self.filenames)) #200 classes, len:8855

        self.number_example = len(self.filenames) #8855

        self.input_ids = input_ids
        self.segments_ids = segments_ids
        self.sentences = sentences

    def load_bbox(self):
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        #
        filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
        df_filenames = pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(filenames), filenames[0])
        #
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()

            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        #
        return filename_bbox

    def load_captions(self, data_dir, filenames):
        all_captions = []
        for i in range(len(filenames)):
            cap_path = '%s/text/%s.txt' % (data_dir, filenames[i])
            with open(cap_path, "r") as f:
                captions = f.read().decode('utf8').split('\n')
                cnt = 0
                for cap in captions:
                    if len(cap) == 0:
                        continue
                    cap = cap.replace("\ufffd\ufffd", " ")
                    # picks out sequences of alphanumeric characters as tokens
                    # and drops everything else
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(cap.lower())
                    # print('tokens', tokens)
                    if len(tokens) == 0:
                        print('cap', cap)
                        continue

                    tokens_new = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0:
                            tokens_new.append(t)
                    all_captions.append(tokens_new)
                    cnt += 1
                    if cnt == self.embeddings_num:
                        break
                if cnt < self.embeddings_num:
                    print('ERROR: the captions for %s less than %d'% (filenames[i], cnt))
        return all_captions

    def build_dictionary(self, train_captions, test_captions):
        word_counts = defaultdict(float)
        captions = train_captions + test_captions
        for sent in captions:
            for word in sent:
                word_counts[word] += 1

        vocab = [w for w in word_counts if word_counts[w] >= 0]

        ixtoword = {}
        ixtoword[0] = '<end>'
        wordtoix = {}
        wordtoix['<end>'] = 0
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1

        train_captions_new = []
        for t in train_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            train_captions_new.append(rev)

        test_captions_new = []
        for t in test_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            test_captions_new.append(rev)

        return [train_captions_new, test_captions_new,
                ixtoword, wordtoix, len(ixtoword)]

    def load_text_data(self, data_dir, split):
        filepath = os.path.join(data_dir, 'captions.pickle')
        train_names = self.load_filenames(data_dir, 'train')
        test_names = self.load_filenames(data_dir, 'test')
        if not os.path.isfile(filepath):
            train_captions = self.load_captions(data_dir, train_names)
            test_captions = self.load_captions(data_dir, test_names)

            train_captions, test_captions, ixtoword, wordtoix, n_words = \
                self.build_dictionary(train_captions, test_captions)
            with open(filepath, 'wb') as f:
                pickle.dump([train_captions, test_captions,
                                ixtoword, wordtoix], f, protocol=2)
                print('Save to: ', filepath)
        else:
            with open(filepath, 'rb') as f:
                x = pickle.load(f)
                train_captions, test_captions = x[0], x[1]
                ixtoword, wordtoix = x[2], x[3]
                del x
                n_words = len(ixtoword)
                print('Load from: ', filepath)
        if split == 'train':
            # a list of list: each list contains
            # the indices of words in a sentence
            captions = train_captions
            filenames = train_names
        else:  # split=='test'
            captions = test_captions
            filenames = test_names
        return filenames, captions, ixtoword, wordtoix, n_words

    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f , encoding = 'latin1')
        else:
            class_id = np.arange(total_num)
        return class_id

    def load_filenames(self, data_dir, split):
        filepath = '%s/%s/filenames.pickle' % (data_dir, split)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f)
            print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        else:
            filenames = []
        return filenames

    def get_caption(self, sent_ix):
        # a list of indices for a sentence
        sent_caption = np.asarray(self.captions[sent_ix]).astype('int64')
        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)
        # pad with 0s (i.e., '<end>')
        x = np.zeros((cfg.TEXT.WORDS_NUM, 1), dtype='int64')
        x_len = num_words
        if num_words <= cfg.TEXT.WORDS_NUM:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:cfg.TEXT.WORDS_NUM]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = cfg.TEXT.WORDS_NUM
        return x, x_len

    def __getitem__(self, index):
        #
        key = self.filenames[index]
        cls_id = self.class_id[index]
        #
        if self.bbox is not None:
            bbox = self.bbox[key]
            data_dir = '%s/CUB_200_2011' % self.data_dir
        else:
            bbox = None
            data_dir = self.data_dir
        #
        img_name = '%s/images/%s.jpg' % (data_dir, key)
        imgs = get_imgs(img_name, self.imsize,
                        bbox, self.transform, normalize=self.norm)
        # random select a sentence
        sent_ix = random.randint(0, self.embeddings_num)
        new_sent_ix = index * self.embeddings_num + sent_ix
        caps, cap_len = self.get_caption(new_sent_ix)

        caps_dec = []
        for word in caps:
            caps_dec.append(self.ixtoword[int(word)])

        return imgs, caps, cap_len, cls_id, key, self.input_ids[index], self.segments_ids[index]


    def __len__(self):
        return len(self.filenames)


class CNN_ENCODER(nn.Module):
    def __init__(self, nef):
        super(CNN_ENCODER, self).__init__()
        if cfg.TRAIN.FLAG:
            self.nef = nef
        else:
            self.nef = 256  # define a uniform ranker
        
        print('CNN_ENCODER')
        model = models.inception_v3()
        print('loaded_inception')
        url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
        model.load_state_dict(model_zoo.load_url(url))
        for param in model.parameters():
            param.requires_grad = False
        print('Load pretrained model from ', url)
        # print(model)

        self.define_module(model)
        self.init_trainable_weights()

    def define_module(self, model):
        self.Conv2d_1a_3x3 = model.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = model.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = model.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = model.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = model.Conv2d_4a_3x3
        self.Mixed_5b = model.Mixed_5b
        self.Mixed_5c = model.Mixed_5c
        self.Mixed_5d = model.Mixed_5d
        self.Mixed_6a = model.Mixed_6a
        self.Mixed_6b = model.Mixed_6b
        self.Mixed_6c = model.Mixed_6c
        self.Mixed_6d = model.Mixed_6d
        self.Mixed_6e = model.Mixed_6e
        self.Mixed_7a = model.Mixed_7a
        self.Mixed_7b = model.Mixed_7b
        self.Mixed_7c = model.Mixed_7c

        self.emb_features = conv1x1(768, self.nef)
        self.emb_cnn_code = nn.Linear(2048, self.nef)

    def init_trainable_weights(self):
        initrange = 0.1
        self.emb_features.weight.data.uniform_(-initrange, initrange)
        self.emb_cnn_code.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        features = None
        # --> fixed-size input: batch x 3 x 299 x 299
        x = nn.Upsample(size=(299, 299), mode='bilinear')(x)
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192

        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288

        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768

        # image region features
        features = x
        # 17 x 17 x 768

        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        # x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048

        # global image features
        cnn_code = self.emb_cnn_code(x)
        # 512
        if features is not None:
            features = self.emb_features(features)
        return features, cnn_code


class BERT_ENCODER(nn.Module):
    def __init__(self):
        super(BERT_ENCODER, self).__init__()

        
        print('Bert_ENCODER')
        
        model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
        for param in model.parameters():
            param.requires_grad = False
        print('Load pretrained model from  BertModel')

        self.define_module(model)
        self.init_trainable_weights()
        self.bert_model = model

    def define_module(self, model):
        self.word_bert_code1 = nn.Linear(3072, 256)
        # self.word_bert_code2 = nn.Linear(2048, 1024)
        # self.word_bert_code3 = nn.Linear(1024, 512)
        # self.word_bert_code4 = nn.Linear(512, 256)
        self.sent_bert_code = nn.Linear(768, 256)
        self.m = nn.Tanh()

    def init_trainable_weights(self):
        initrange = 0.1
        self.word_bert_code1.weight.data.uniform_(-0.1, 0.1)
        # self.word_bert_code2.weight.data.uniform_(0, 0)
        # self.word_bert_code3.weight.data.uniform_(0, 0)
        # self.word_bert_code4.weight.data.uniform_(0, 0)
        self.sent_bert_code.weight.data.uniform_(-initrange, initrange)

    def forward(self,  b_input_ids, b_segments_ids):
        
        outputs = self.bert_model(b_input_ids, b_segments_ids)
        hidden_states = outputs[2]

        word_embedding = torch.stack(hidden_states, dim=0)
        word_embedding = word_embedding.permute(1,2,0,3)

        all_samples= []
        for sample in word_embedding:
            token_vecs_cat = []
            for token in sample:
                cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
                token_vecs_cat.append(cat_vec)
            all_samples.append(token_vecs_cat)

        a = []
        for i in range (len(all_samples)):
            a.append(torch.stack(all_samples[i]))

        word_embedding = torch.stack(a)


        word_embedding= self.m(self.word_bert_code1(word_embedding))
        # word_embedding= self.m(self.word_bert_code2(word_embedding))
        # word_embedding= self.m(self.word_bert_code3(word_embedding))
        # word_embedding= self.m(self.word_bert_code4(word_embedding))

        word_embedding = word_embedding.permute(0,2,1)





        token_vecs = hidden_states[-2]

        # Calculate the average of all 18 token vectors.
        sentence_embedding = torch.mean(token_vecs, dim=1)
        sentence_embedding= self.sent_bert_code(sentence_embedding)


        return  word_embedding, sentence_embedding


def drawCaption(convas, captions, ixtoword, vis_size, off1=2, off2=2):
    
    FONT_MAX = 50

    num = captions.size(0)
    img_txt = Image.fromarray(convas)
    # get a font
    # fnt = None  # ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 50)
    print ("CURRENT WORKING DIRCTORY : " , os.getcwd())
    fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 50)
    # get a drawing context
    d = ImageDraw.Draw(img_txt)
    sentence_list = []
    for i in range(num):
        cap = captions[i].data.cpu().numpy()
        sentence = []
        for j in range(len(cap)):
            if cap[j] == 0:
                break
            word = ixtoword[cap[j]].encode('ascii', 'ignore').decode('ascii')
            d.text(((j + off1) * (vis_size + off2), i * FONT_MAX), '%d:%s' % (j, word[:6]),
                   font=fnt, fill=(255, 255, 255, 255))
            sentence.append(word)
        sentence_list.append(sentence)
    return img_txt, sentence_list

def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim.
    """
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

def sent_loss(cnn_code, rnn_code, labels, class_ids, batch_size, eps=1e-8):
    # ### Mask mis-match samples  ###
    # that come from the same class as the real sample ###
    masks = []
    if class_ids is not None:
        for i in range(batch_size):
            mask = (class_ids == class_ids[i]).astype(np.uint8)
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))
        masks = np.concatenate(masks, 0)
        # masks: batch_size x batch_size
        masks = torch.BoolTensor(masks)
        if cfg.CUDA:
            masks = masks.cuda()

    # --> seq_len x batch_size x nef
    if cnn_code.dim() == 2:
        cnn_code = cnn_code.unsqueeze(0)
        rnn_code = rnn_code.unsqueeze(0)

    # cnn_code_norm / rnn_code_norm: seq_len x batch_size x 1
    cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True)
    rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True)
    # scores* / norm*: seq_len x batch_size x batch_size
    scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
    norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
    scores0 = scores0 / norm0.clamp(min=eps) * cfg.TRAIN.SMOOTH.GAMMA3

    # --> batch_size x batch_size
    scores0 = scores0.squeeze()
    if class_ids is not None:
        scores0.data.masked_fill_(masks, -float('inf'))
    scores1 = scores0.transpose(0, 1)
    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(scores0, labels)
        loss1 = nn.CrossEntropyLoss()(scores1, labels)
    else:
        loss0, loss1 = None, None
    return loss0, loss1


def words_loss(img_features, words_emb, labels, cap_lens, class_ids, batch_size):
    """
        words_emb(query): batch x nef x seq_len
        img_features(context): batch x nef x 17 x 17
    """
    masks = []
    att_maps = []
    similarities = []
    cap_lens = cap_lens.data.tolist()

    for i in range(batch_size):
    
        if class_ids is not None:
            mask = (class_ids == class_ids[i]).astype(np.uint8)
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))
        # Get the i-th text description
        words_num = cap_lens[i]
        # -> 1 x nef x words_num
        word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()
        # -> batch_size x nef x words_num
        word = word.repeat(batch_size, 1, 1)
        # batch x nef x 17*17
        context = img_features
        """
            word(query): batch x nef x words_num
            context: batch x nef x 17 x 17
            weiContext: batch x nef x words_num
            attn: batch x words_num x 17 x 17
        """
        weiContext, attn = func_attention(word, context, cfg.TRAIN.SMOOTH.GAMMA1)
        att_maps.append(attn[i].unsqueeze(0).contiguous())
        # --> batch_size x words_num x nef
        word = word.transpose(1, 2).contiguous()
        weiContext = weiContext.transpose(1, 2).contiguous()
        # --> batch_size*words_num x nef
        word = word.view(batch_size * words_num, -1)
        weiContext = weiContext.view(batch_size * words_num, -1)
        #
        # -->batch_size*words_num
        row_sim = cosine_similarity(word, weiContext)
        # --> batch_size x words_num
        row_sim = row_sim.view(batch_size, words_num)

        # Eq. (10)
        row_sim.mul_(cfg.TRAIN.SMOOTH.GAMMA2).exp_()
        row_sim = row_sim.sum(dim=1, keepdim=True)
        row_sim = torch.log(row_sim)

        # --> 1 x batch_size
        # similarities(i, j): the similarity between the i-th image and the j-th text description
        similarities.append(row_sim)


    # batch_size x batch_size
    similarities = torch.cat(similarities, 1)
    if class_ids is not None:
        masks = np.concatenate(masks, 0)
        # masks: batch_size x batch_size
        masks = torch.BoolTensor(masks)
        if cfg.CUDA:
            masks = masks.cuda()

    similarities = similarities * cfg.TRAIN.SMOOTH.GAMMA3
    if class_ids is not None:
        similarities.data.masked_fill_(masks, -float('inf'))
    similarities1 = similarities.transpose(0, 1)
    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(similarities, labels)
        loss1 = nn.CrossEntropyLoss()(similarities1, labels)
    else:
        loss0, loss1 = None, None
    return loss0, loss1, att_maps

def func_attention(query, context, gamma1):
    """
    query: batch x ndf x queryL
    context: batch x ndf x ih x iw (sourceL=ihxiw)
    mask: batch_size x sourceL
    """
    batch_size, queryL = query.size(0), query.size(2)
    ih, iw = context.size(2), context.size(3)
    sourceL = ih * iw

    # --> batch x sourceL x ndf
    context = context.view(batch_size, -1, sourceL)
    contextT = torch.transpose(context, 1, 2).contiguous()

    # Get attention
    # (batch x sourceL x ndf)(batch x ndf x queryL)
    # -->batch x sourceL x queryL
    attn = torch.bmm(contextT, query) # Eq. (7) in AttnGAN paper
    # --> batch*sourceL x queryL
    attn = attn.view(batch_size*sourceL, queryL)
    attn = nn.Softmax()(attn)  # Eq. (8)

    # --> batch x sourceL x queryL
    attn = attn.view(batch_size, sourceL, queryL)
    # --> batch*queryL x sourceL
    attn = torch.transpose(attn, 1, 2).contiguous()
    attn = attn.view(batch_size*queryL, sourceL)
    #  Eq. (9)
    attn = attn * gamma1
    attn = nn.Softmax()(attn)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> batch x sourceL x queryL
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # (batch x ndf x sourceL)(batch x sourceL x queryL)
    # --> batch x ndf x queryL
    weightedContext = torch.bmm(context, attnT)

    return weightedContext, attn.view(batch_size, -1, ih, iw)


def train(dataloader, cnn_model, bert_encoder, batch_size, labels, optimizer, epoch, ixtoword, image_dir):
    train_function_start_time = time.time()
    cnn_model.train() #Sets the module in training mode.
    #rnn_model.train() #Sets the module in training mode.
    bert_encoder.train()
    
    s_total_loss0 = 0
    s_total_loss1 = 0
    w_total_loss0 = 0
    w_total_loss1 = 0
    
    # print("keyword |||||||||||||||||||||||||||||||")
    # print("len(dataloader) : " , len(dataloader) )
    # print(" count = " ,  (epoch + 1) * len(dataloader)  )
    # print("keyword |||||||||||||||||||||||||||||||")
    count = (epoch + 1) * len(dataloader)
    start_time = time.time()


    for step, data in enumerate(dataloader, 0):
    
    
        # Loading the first batch (number of batches/steps in an epoch is 183)
        bert_encoder.zero_grad()
        cnn_model.zero_grad()

        imgs, captions, cap_lens, class_ids, keys, b_input_ids, b_segments_ids = prepare_data(data)


        # words_features: batch_size x 256 x 17 x 17 ==> # image region features
        # sent_code: batch_size x 256                ==> # global image features
        words_features, sent_code = cnn_model(imgs[-1])
        # --> batch_size x nef x 17*17
        nef, att_sze = words_features.size(1), words_features.size(2)# 256, 17(16th of the whole image)
        # words_features = words_features.view(batch_size, nef, -1)

        #hidden = rnn_model.init_hidden(batch_size) # A tuple of 2 zero tensor of torch.Size([2, 48, 128])
        # words_emb: batch_size x nef x seq_len
        # sent_emb: batch_size x nef
        #words_emb, sent_emb = rnn_model(captions, cap_lens, hidden)

        #-------------------------------------------------------------------------------
                #---------------------------------------------------------
                        #-----------------------------------
        words_emb, sent_emb =bert_encoder(b_input_ids, b_segments_ids)
                        #-----------------------------------
                #---------------------------------------------------------
        #-------------------------------------------------------------------------------

        w_loss0, w_loss1, attn_maps = words_loss(words_features, words_emb, labels, cap_lens, class_ids, batch_size)
        w_total_loss0 += w_loss0.data
        w_total_loss1 += w_loss1.data
        loss = w_loss0 + w_loss1

        s_loss0, s_loss1 = sent_loss(sent_code, sent_emb, labels, class_ids, batch_size)
        loss += s_loss0 + s_loss1
        s_total_loss0 += s_loss0.data
        s_total_loss1 += s_loss1.data
        #
        loss.backward()
        #
        # `clip_grad_norm` helps prevent
        # the exploding gradient problem in RNNs / LSTMs.
        #torch.nn.utils.clip_grad_norm(bert_encoder.parameters(), cfg.TRAIN.RNN_GRAD_CLIP)
        optimizer.step()


        if step % UPDATE_INTERVAL == 0:
            print ('Max of word features', words_emb.max())
            print ('Min of word features', words_emb.min())
            
            count = epoch * len(dataloader) + step

            # print ("====================================================")
            # print ("s_total_loss0 : " , s_total_loss0)
            # print ("s_total_loss0.item() : " , s_total_loss0.item())
            # print ("UPDATE_INTERVAL : " , UPDATE_INTERVAL)
            print ("s_total_loss0.item()/UPDATE_INTERVAL : " , s_total_loss0.item()/UPDATE_INTERVAL)
            print ("s_total_loss1.item()/UPDATE_INTERVAL : " , s_total_loss1.item()/UPDATE_INTERVAL)
            print ("w_total_loss0.item()/UPDATE_INTERVAL : " , w_total_loss0.item()/UPDATE_INTERVAL)
            print ("w_total_loss1.item()/UPDATE_INTERVAL : " , w_total_loss1.item()/UPDATE_INTERVAL)
            # print ("s_total_loss0/UPDATE_INTERVAL : " , s_total_loss0/UPDATE_INTERVAL)
            # print ("=====================================================")
            s_cur_loss0 = s_total_loss0.item() / UPDATE_INTERVAL
            s_cur_loss1 = s_total_loss1.item() / UPDATE_INTERVAL

            w_cur_loss0 = w_total_loss0.item() / UPDATE_INTERVAL
            w_cur_loss1 = w_total_loss1.item() / UPDATE_INTERVAL

            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                    's_loss {:5.2f} {:5.2f} | '
                    'w_loss {:5.2f} {:5.2f}'
                    .format(epoch, step, len(dataloader),
                          elapsed * 1000. / UPDATE_INTERVAL,
                            s_cur_loss0, s_cur_loss1,
                            w_cur_loss0, w_cur_loss1))
            s_total_loss0 = 0
            s_total_loss1 = 0
            w_total_loss0 = 0
            w_total_loss1 = 0
            start_time = time.time()
            # attention Maps
            #Save image only every 8 epochs && Save it to The Drive
            if (epoch % 20 == 0 and epoch > 0):
                print("bulding images")
                img_set, _ = build_super_images(imgs[-1].cpu(), captions, ixtoword, attn_maps, att_sze)
                if img_set is not None:
                    im = Image.fromarray(img_set)
                    fullpath = '%s/attention_maps%d.png' % (image_dir, step)
                    im.save(fullpath)
                    mydriveimg = '/content/drive/My Drive/BertCUBImage'
                    drivepath = '%s/attention_maps%d.png' % (mydriveimg, epoch)
                    im.save(drivepath)
    print("keyTime |||||||||||||||||||||||||||||||")
    print("train_function_time : " , time.time() - train_function_start_time)
    print("KeyTime |||||||||||||||||||||||||||||||")
    return count


def evaluate(dataloader, cnn_model, bert_encoder, batch_size):
    cnn_model.eval()
    bert_encoder.eval()
    s_total_loss = 0
    w_total_loss = 0
    for step, data in enumerate(dataloader, 0):
        real_imgs, captions, cap_lens, class_ids, keys, b_input_ids, b_segments_ids = prepare_data(data)

        words_features, sent_code = cnn_model(real_imgs[-1])
        # nef = words_features.size(1)
        # words_features = words_features.view(batch_size, nef, -1)
        words_emb, sent_emb =bert_encoder(b_input_ids, b_segments_ids)



        w_loss0, w_loss1, attn = words_loss(words_features, words_emb, labels, cap_lens, class_ids, batch_size)
        w_total_loss += (w_loss0 + w_loss1).data

        s_loss0, s_loss1 = sent_loss(sent_code, sent_emb, labels, class_ids, batch_size)
        s_total_loss += (s_loss0 + s_loss1).data

        if step == 50:
            break

    s_cur_loss = s_total_loss.item() / step
    w_cur_loss = w_total_loss.item() / step

    return s_cur_loss, w_cur_loss


def build_models():
    # build model ############################################################
    #text_encoder = RNN_ENCODER(dataset.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
    '''
    RNN_ENCODER(
    (encoder): Embedding(5450, 300)
    (drop): Dropout(p=0.5, inplace=False)
    (rnn): LSTM(300, 128, batch_first=True, dropout=0.5, bidirectional=True))
    '''
    print('build_models')
    image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
    bert_encoder = BERT_ENCODER()

    labels = Variable(torch.LongTensor(range(batch_size)))
    '''
    A tensor of [0,1,2,3,...,47]
    '''
    start_epoch = 0
    if cfg.TRAIN.NET_E != '':
        state_dict = torch.load(cfg.TRAIN.NET_E)
        bert_encoder.load_state_dict(state_dict)
        print('Load ', cfg.TRAIN.NET_E)
        #
        name = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
        state_dict = torch.load(name)
        image_encoder.load_state_dict(state_dict)
        print('Load ', name)

        istart = cfg.TRAIN.NET_E.rfind('_') + 8
        iend = cfg.TRAIN.NET_E.rfind('.')
        start_epoch = cfg.TRAIN.NET_E[istart:iend]
        start_epoch = int(start_epoch) + 1
        print('start_epoch', start_epoch)
    if cfg.CUDA:
        #text_encoder = text_encoder.cuda()
        image_encoder = image_encoder.cuda()
        labels = labels.cuda()
        bert_encoder = bert_encoder.cuda()

    return bert_encoder, image_encoder, labels, start_epoch


def build_models2():
    # build model ############################################################
    #text_encoder = RNN_ENCODER(dataset.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
    '''
    RNN_ENCODER(
    (encoder): Embedding(5450, 300)
    (drop): Dropout(p=0.5, inplace=False)
    (rnn): LSTM(300, 128, batch_first=True, dropout=0.5, bidirectional=True))
    '''
    print('build_models')
    image_encoder = myimage_encoder
    bert_encoder = BERT_ENCODER()

    labels = Variable(torch.LongTensor(range(batch_size)))
    '''
    A tensor of [0,1,2,3,...,47]
    '''
    start_epoch = 0
    if cfg.CUDA:
        #text_encoder = text_encoder.cuda()
        #image_encoder = image_encoder.cuda()
        labels = labels.cuda()
        bert_encoder = bert_encoder.cuda()

    return bert_encoder,image_encoder, labels, start_epoch


__name__ = "__main__"
if __name__ == "__main__":
    print('Using config:')
    pprint.pprint(cfg)

    UPDATE_INTERVAL = 60

    ##########################################################################
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '../output/%s_%s_%s' % (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    model_dir = os.path.join(output_dir, 'Model')
    image_dir = os.path.join(output_dir, 'Image')
    mkdir_p(model_dir)
    mkdir_p(image_dir)

    torch.cuda.set_device(cfg.GPU_ID)
    cudnn.benchmark = True

    # Get data loader ##################################################
    imsize = 299
    batch_size = 48

    image_transform = transforms.Compose([transforms.Scale(355), transforms.RandomCrop(imsize), transforms.RandomHorizontalFlip()])
    
    dataset = TextDataset(cfg.DATA_DIR, 'train', base_size=cfg.TREE.BASE_SIZE, transform=image_transform, input_ids=input_ids, segments_ids=segments_ids, sentences=sentences )
    print(dataset.n_words, dataset.embeddings_num)
    assert dataset

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=int(cfg.WORKERS))
    #using prepare data functiont this dataloader yieldes:
    #imgs: a list of 1 tensor of size torch.Size([48, 3, 299, 299])
    #captons: a  tensor of size torch.Size([48, 18]), shorter filled with end words converted by word to index
    #cap_lens: a  tensor of size torch.Size([48]) , acual caps lens order from big to small (max is 18)
    #class_ids: a 48 ints list in range 0-200 of the classes
    #keys: a 48 string list  of the classes classes nammes crosspondening to the class_ids
    

    # # validation data #
    dataset_val = TextDataset(cfg.DATA_DIR, 'test', base_size=cfg.TREE.BASE_SIZE,transform=image_transform, input_ids=input_ids, segments_ids=segments_ids, sentences=sentences)
    print(dataset_val.n_words, dataset_val.embeddings_num)
    assert dataset_val
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, drop_last=True,shuffle=True, num_workers=int(cfg.WORKERS))

    # Train ##############################################################
    bert_encoder, image_encoder, labels, start_epoch = build_models()
    #bert_encoder,image_encoder, labels, start_epoch = build_models2()
    para = []
    
    for v in bert_encoder.parameters(): # 4 parameters
        if v.requires_grad:
            para.append(v)
    
    for v in image_encoder.parameters(): # 3 parameters
        if v.requires_grad:
            para.append(v)

    print ('requires_grad =' , len(para))
    # optimizer = optim.Adam(para, lr=cfg.TRAIN.ENCODER_LR, betas=(0.5, 0.999))
    # At any point you can hit Ctrl + C to break out of training early.

    try:
        lr = cfg.TRAIN.ENCODER_LR #0.002
        lr = 0.02
        print("keyword |||||||||||||||||||||||||||||||")
        print("Start_epoch : " , start_epoch)
        print("cfg.TRAIN.MAX_EPOCH : " , cfg.TRAIN.MAX_EPOCH )
        print("keyword |||||||||||||||||||||||||||||||")


        for epoch in range(start_epoch, cfg.TRAIN.MAX_EPOCH):
            
            one_epoch_start_time = time.time()
            optimizer = optim.Adam(para, lr=lr, betas=(0.5, 0.999))
            epoch_start_time = time.time()
            count = train(dataloader, image_encoder, bert_encoder, batch_size, labels, optimizer, epoch, dataset.ixtoword, image_dir)
            print('-' * 89)
            if len(dataloader_val) > 0:
                s_loss, w_loss = evaluate(dataloader_val, image_encoder, bert_encoder, batch_size)
                print('| end epoch {:3d} | valid loss ''{:5.2f} {:5.2f} | lr {:.5f}|'.format(epoch, s_loss, w_loss, lr))
            print('-' * 89)
            if lr > 0.0002 : #cfg.TRAIN.ENCODER_LR/10.:
                lr *= 0.98

            print("keyTime |||||||||||||||||||||||||||||||")
            print("one_epoch_time : " , time.time() - one_epoch_start_time)
            print("KeyTime |||||||||||||||||||||||||||||||")

            if (epoch % 20 == 0 or epoch == cfg.TRAIN.MAX_EPOCH or epoch == cfg.TRAIN.MAX_EPOCH-1 ):
                mydrivemodel = '/content/drive/My Drive/BertCUBModel'
                torch.save(image_encoder.state_dict(), '%s/image_encoder%d.pth' % (model_dir, epoch))
                torch.save(image_encoder.state_dict(), '%s/image_encoder%d.pth' % (mydrivemodel, epoch))
                torch.save(bert_encoder.state_dict(), '%s/text_encoder%d.pth' % (model_dir, epoch))
                torch.save(bert_encoder.state_dict(), '%s/text_encoder%d.pth' % (mydrivemodel, epoch))
                print('Save G/Ds models.')


    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

