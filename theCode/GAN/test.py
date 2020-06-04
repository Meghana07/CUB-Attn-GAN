####Configerations Part###
DATA_DIR = "../data/birds"
EMBEDDING_DIM = 256
NET_G = "netG_epoch_600.pth"
WORDS_NUM = 18
RNN_TYPE = 'LSTM'
NET_E = "text_encoder599.pth"
B_DCGAN = False
GF_DIM = 128
CONDITION_DIM = 100
###########

###Imports Part###
import numpy as np
import pickle
from nltk.tokenize import RegexpTokenizer
import os
from collections import defaultdict
import torch
import torch.nn as nn
############

###Generate Examples###

filepath = 'captions.pickle'  # Load captions.pickle contain four parts :
#captions[0] : train_captions, 88550 caption each of different length of words (9-25),each word represented by unique index
#captions[1] : test_captions , 29330 caption each of different length of words (9-25),each word represented by unique index
#captions[2] : index to word dictionary that maps 5450 indexes[key] to words[value]
#captions[3] : word to index dictionary that maps 5450 words[key] to indexes[value]

with open(filepath, 'rb') as f:
    x = pickle.load(f)
    wordtoix = x[3]
    del x
    n_words = len(wordtoix)
    print("number of words : ", n_words)

algo = ''
filepath = 'example_filenames.txt'
#a group of example filenames, 24 names for 24 file, each file with a number of captions

data_dic = {}  # dictionary used to generate images from captions
#key : name of file from which we got captions
#value: [padded-sorted_based_on_length-indexed captions, original length of each(before padding, indexes to order based on length)

with open(filepath, "r") as f:
    filenames = f.read().split('\n')
    for name in filenames:
        if name == "example_captions":  #Keep this way until you download the rest of the file captions
            if len(name) == 0:
                continue
            filepath = '%s.txt' % (name)
            with open(filepath, "r") as f:
                print('Load from:', name)
                sentences = f.read().split('\n')
                # split your text file of 16 captions to a list of 16 string entries
                print("sentences : ", sentences)
                captions = []
                cap_lens = []
                for sent in sentences:
                    if len(sent) == 0:
                        continue
                    sent = sent.replace("\ufffd\ufffd", " ")
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(sent.lower())
                    #convert a single sentence(string) to list of tokens(words)=>result in a list of string entries that are word that make up the original sentence(caption)
                    print("tokens : ", tokens)
                    if len(tokens) == 0:
                        print('sent', sent)
                        continue

                    rev = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0 and t in wordtoix:
                            rev.append(wordtoix[t])
                            #convert the list of words to a list crosspending indexes
                    captions.append(rev)  # all captions in the file
                    cap_lens.append(len(rev))
                    # the length(number of words/tokens) in each caption
            max_len = np.max(cap_lens)  # used to pad shorter captions
            print("captions : ", captions)
            print("number of captions : ", len(captions))
            print("max_len : ", max_len)
            print("cap_lens : ", cap_lens)

            sorted_indices = np.argsort(cap_lens)[::-1]
            # Returns the indices that would sort the array of lengths.
            print("sorted_indices : ", sorted_indices)
            cap_lens = np.asarray(cap_lens)
            cap_lens = cap_lens[sorted_indices]
            # sort the array of lengths using the sotring indices
            print("sorted cap_lens : ", cap_lens)
            cap_array = np.zeros(
                (len(captions), max_len), dtype='int64'
            )  #placeholder for the padded sorted array caption
            for i in range(len(captions)):
                idx = sorted_indices[i]
                cap = captions[idx]
                c_len = len(cap)
                cap_array[i, :c_len] = cap
            print("padded sorted array caption : ", cap_array)
            key = name[(name.rfind('/') + 1):]
            print("name : ", name)
            print("key : ", key)
            data_dic[key] = [cap_array, cap_lens, sorted_indices]
            print("data_dic", data_dic)


def gen_example(self, data_dic):
    if NET_G == '':
        print('Error: the path for models is not found!')
    else:
        # Build and load the generator
        text_encoder = RNN_ENCODER(n_words, nhidden=EMBEDDING_DIM)
        state_dict = torch.load(NET_E,
                                map_location=lambda storage, loc: storage)
        text_encoder.load_state_dict(state_dict)
        print('Load text encoder from:', NET_E)
        text_encoder = text_encoder.cuda()
        text_encoder.eval()

        # the path to save generated images
        if cfg.GAN.B_DCGAN:
            netG = G_DCGAN()
        else:
            netG = G_NET()
        s_tmp = cfg.TRAIN.NET_G[:cfg.TRAIN.NET_G.rfind('.pth')]
        model_dir = cfg.TRAIN.NET_G
        state_dict = \
            torch.load(model_dir, map_location=lambda storage, loc: storage)
        netG.load_state_dict(state_dict)
        print('Load G from: ', model_dir)
        netG.cuda()
        netG.eval()
        for key in data_dic:
            save_dir = '%s/%s' % (s_tmp, key)
            mkdir_p(save_dir)
            captions, cap_lens, sorted_indices = data_dic[key]

            batch_size = captions.shape[0]
            nz = cfg.GAN.Z_DIM
            captions = Variable(torch.from_numpy(captions), volatile=True)
            cap_lens = Variable(torch.from_numpy(cap_lens), volatile=True)

            captions = captions.cuda()
            cap_lens = cap_lens.cuda()
            for i in range(1):  # 16
                noise = Variable(torch.FloatTensor(batch_size, nz),
                                 volatile=True)
                noise = noise.cuda()
                #######################################################
                # (1) Extract text embeddings
                ######################################################
                hidden = text_encoder.init_hidden(batch_size)
                # words_embs: batch_size x nef x seq_len
                # sent_emb: batch_size x nef
                words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                mask = (captions == 0)
                #######################################################
                # (2) Generate fake images
                ######################################################
                noise.data.normal_(0, 1)
                fake_imgs, attention_maps, _, _ = netG(noise, sent_emb,
                                                       words_embs, mask)
                # G attention
                cap_lens_np = cap_lens.cpu().data.numpy()
                for j in range(batch_size):
                    save_name = '%s/%d_s_%d' % (save_dir, i, sorted_indices[j])
                    for k in range(len(fake_imgs)):
                        im = fake_imgs[k][j].data.cpu().numpy()
                        im = (im + 1.0) * 127.5
                        im = im.astype(np.uint8)
                        # print('im', im.shape)
                        im = np.transpose(im, (1, 2, 0))
                        # print('im', im.shape)
                        im = Image.fromarray(im)
                        fullpath = '%s_g%d.png' % (save_name, k)
                        im.save(fullpath)

                    for k in range(len(attention_maps)):
                        if len(fake_imgs) > 1:
                            im = fake_imgs[k + 1].detach().cpu()
                        else:
                            im = fake_imgs[0].detach().cpu()
                        attn_maps = attention_maps[k]
                        att_sze = attn_maps.size(2)
                        img_set, sentences = \
                            build_super_images2(im[j].unsqueeze(0),
                                                captions[j].unsqueeze(0),
                                                [cap_lens_np[j]], self.ixtoword,
                                                [attn_maps[j]], att_sze)
                        if img_set is not None:
                            im = Image.fromarray(img_set)
                            fullpath = '%s_a%d.png' % (save_name, k)
                            im.save(fullpath)


###Class of Models###
class RNN_ENCODER(nn.Module):
    def __init__(self,
                ntoken,
                ninput=300,
                drop_prob=0.5,
                nhidden=128,
                nlayers=1,
                bidirectional=True):
        super(RNN_ENCODER, self).__init__()
        self.n_steps = WORDS_NUM
        self.ntoken = ntoken  # size of the dictionary that maps words to unique indexes
        self.ninput = ninput  # size of each embedding vector
        self.drop_prob = drop_prob  # probability of an element to be zeroed
        self.nlayers = nlayers  # Number of recurrent layers
        self.bidirectional = bidirectional
        self.rnn_type = RNN_TYPE
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        # number of features in the hidden state
        self.nhidden = nhidden // self.num_directions

        self.define_module()
        self.init_weights()

    def define_module(self):
        self.encoder = nn.Embedding(self.ntoken, self.ninput)
        self.drop = nn.Dropout(self.drop_prob)
        if self.rnn_type == 'LSTM':
            # dropout: If non-zero, introduces a dropout layer on
            # the outputs of each RNN layer except the last layer
            self.rnn = nn.LSTM(self.ninput,
                               self.nhidden,
                               self.nlayers,
                               batch_first=True,
                               dropout=self.drop_prob,
                               bidirectional=self.bidirectional)
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(self.ninput,
                              self.nhidden,
                              self.nlayers,
                              batch_first=True,
                              dropout=self.drop_prob,
                              bidirectional=self.bidirectional)
        else:
            raise NotImplementedError

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        # Do not need to initialize RNN parameters, which have been initialized
        # http://pytorch.org/docs/master/_modules/torch/nn/modules/rnn.html#LSTM
        # self.decoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.fill_(0)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(
                weight.new(self.nlayers * self.num_directions, bsz,
                           self.nhidden).zero_()),
                    Variable(
                        weight.new(self.nlayers * self.num_directions, bsz,
                                   self.nhidden).zero_()))
        else:
            return Variable(
                weight.new(self.nlayers * self.num_directions, bsz,
                           self.nhidden).zero_())

    def forward(self, captions, cap_lens, hidden, mask=None):
        # input: torch.LongTensor of size batch x n_steps
        # --> emb: batch x n_steps x ninput
        emb = self.drop(self.encoder(captions))
        #
        # Returns: a PackedSequence object
        cap_lens = cap_lens.data.tolist()
        emb = pack_padded_sequence(emb, cap_lens, batch_first=True)
        # #hidden and memory (num_layers * num_directions, batch, hidden_size):
        # tensor containing the initial hidden state for each element in batch.
        # #output (batch, seq_len, hidden_size * num_directions)
        # #or a PackedSequence object:
        # tensor containing output features (h_t) from the last layer of RNN
        output, hidden = self.rnn(emb, hidden)
        # PackedSequence object
        # --> (batch, seq_len, hidden_size * num_directions)
        output = pad_packed_sequence(output, batch_first=True)[0]
        # output = self.drop(output)
        # --> batch x hidden_size*num_directions x seq_len
        words_emb = output.transpose(1, 2)
        # --> batch x num_directions*hidden_size
        if self.rnn_type == 'LSTM':
            sent_emb = hidden[0].transpose(0, 1).contiguous()
        else:
            sent_emb = hidden.transpose(0, 1).contiguous()
        sent_emb = sent_emb.view(-1, self.nhidden * self.num_directions)
        return words_emb, sent_emb


class CA_NET(nn.Module):
    # some code is modified from vae examples
    # (https://github.com/pytorch/examples/blob/master/vae/main.py)
    def __init__(self):
        super(CA_NET, self).__init__()
        self.t_dim = cfg.TEXT.EMBEDDING_DIM
        self.c_dim = cfg.GAN.CONDITION_DIM
        self.fc = nn.Linear(self.t_dim, self.c_dim * 4, bias=True)
        self.relu = GLU()


class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc / 2)
        return x[:, :nc] * F.sigmoid(x[:, nc:])


class G_NET(nn.Module):
    def __init__(self):
        super(G_NET, self).__init__()
        ngf = cfg.GAN.GF_DIM
        nef = cfg.TEXT.EMBEDDING_DIM
        ncf = cfg.GAN.CONDITION_DIM
        self.ca_net = CA_NET()

        if cfg.TREE.BRANCH_NUM > 0:
            self.h_net1 = INIT_STAGE_G(ngf * 16, ncf)
            self.img_net1 = GET_IMAGE_G(ngf)
        # gf x 64 x 64
        if cfg.TREE.BRANCH_NUM > 1:
            self.h_net2 = NEXT_STAGE_G(ngf, nef, ncf)
            self.img_net2 = GET_IMAGE_G(ngf)
        if cfg.TREE.BRANCH_NUM > 2:
            self.h_net3 = NEXT_STAGE_G(ngf, nef, ncf)
            self.img_net3 = GET_IMAGE_G(ngf)

    def forward(self, z_code, sent_emb, word_embs, mask):
        """
            :param z_code: batch x cfg.GAN.Z_DIM
            :param sent_emb: batch x cfg.TEXT.EMBEDDING_DIM
            :param word_embs: batch x cdf x seq_len
            :param mask: batch x seq_len
            :return:
        """
        fake_imgs = []
        att_maps = []
        c_code, mu, logvar = self.ca_net(sent_emb)

        if cfg.TREE.BRANCH_NUM > 0:
            h_code1 = self.h_net1(z_code, c_code)
            fake_img1 = self.img_net1(h_code1)
            fake_imgs.append(fake_img1)
        if cfg.TREE.BRANCH_NUM > 1:
            h_code2, att1 = \
                self.h_net2(h_code1, c_code, word_embs, mask)
            fake_img2 = self.img_net2(h_code2)
            fake_imgs.append(fake_img2)
            if att1 is not None:
                att_maps.append(att1)
        if cfg.TREE.BRANCH_NUM > 2:
            h_code3, att2 = \
                self.h_net3(h_code2, c_code, word_embs, mask)
            fake_img3 = self.img_net3(h_code3)
            fake_imgs.append(fake_img3)
            if att2 is not None:
                att_maps.append(att2)

        return fake_imgs, att_maps, mu, logvar


gen_example(data_dic)

import os

from collections import defaultdict

#def build_dictionary(self, train_captions, test_captions):
#    word_counts = defaultdict(float)
#     captions = train_captions + test_captions
#     for sent in captions:
#         for word in sent:
#             word_counts[word] += 1

#     vocab = [w for w in word_counts if word_counts[w] >= 0]

#     ixtoword = {}
#     ixtoword[0] = '<end>'
#     wordtoix = {}
#     wordtoix['<end>'] = 0
#     ix = 1
#     for w in vocab:
#         wordtoix[w] = ix
#         ixtoword[ix] = w
#         ix += 1

#     train_captions_new = []
#     for t in train_captions:
#         rev = []
#         for w in t:
#             if w in wordtoix:
#                 rev.append(wordtoix[w])
#         # rev.append(0)  # do not need '<end>' token
#         train_captions_new.append(rev)

#     test_captions_new = []
#     for t in test_captions:
#         rev = []
#         for w in t:
#             if w in wordtoix:
#                 rev.append(wordtoix[w])
#         # rev.append(0)  # do not need '<end>' token
#         test_captions_new.append(rev)

#     return [
#         train_captions_new, test_captions_new, ixtoword, wordtoix,
#         len(ixtoword)
#     ]
