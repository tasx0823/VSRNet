import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm  # clip_grad_norm_ for 0.4.0, clip_grad_norm for 0.3.1
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
from loss import TripletLoss
from basic.bigfile import BigFile



def get_we_parameter(vocab, w2v_file):
    w2v_reader = BigFile(w2v_file)
    ndims = w2v_reader.ndims

    we = []
    # we.append([0]*ndims)
    for i in range(len(vocab)):
        try:
            vec = w2v_reader.read_one(vocab.idx2word[i])
        except:
            vec = np.random.uniform(-1, 1, ndims)
        we.append(vec)
    print('getting pre-trained parameter for word embedding initialization', np.shape(we)) 
    return np.array(we)

def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X

def xavier_init_fc(fc):
    """Xavier initialization for the fully connected layer
    """
    r = np.sqrt(6.) / np.sqrt(fc.in_features +
                             fc.out_features)
    fc.weight.data.uniform_(-r, r)
    fc.bias.data.fill_(0)



class MFC(nn.Module):
    """
    Multi Fully Connected Layers
    """
    def __init__(self, fc_layers, dropout, have_dp=True, have_bn=False, have_last_bn=False):
        super(MFC, self).__init__()
        # fc layers
        self.n_fc = len(fc_layers)
        if self.n_fc > 1:
            if self.n_fc > 1:
                self.fc1 = nn.Linear(fc_layers[0], fc_layers[1])

            # dropout
            self.have_dp = have_dp
            if self.have_dp:
                self.dropout = nn.Dropout(p=dropout)

            # batch normalization
            self.have_bn = have_bn
            self.have_last_bn = have_last_bn
            if self.have_bn:
                if self.n_fc == 2 and self.have_last_bn:
                    self.bn_1 = nn.BatchNorm1d(fc_layers[1])

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        if self.n_fc > 1:
            xavier_init_fc(self.fc1)

    def forward(self, inputs):

        if self.n_fc <= 1:
            features = inputs

        elif self.n_fc == 2:
            features = self.fc1(inputs)
            # batch noarmalization
            if self.have_bn and self.have_last_bn:
                features = self.bn_1(features)
            if self.have_dp:
                features = self.dropout(features)

        return features



class Video_multilevel_encoding(nn.Module):
    """
    Section 3.1. Video-side Multi-level Encoding
    """
    def __init__(self, opt):
        super(Video_multilevel_encoding, self).__init__()

        self.rnn_output_size = opt.visual_rnn_size*2
        self.dropout = nn.Dropout(p=opt.dropout)
        self.visual_norm = opt.visual_norm
        self.concate = opt.concate

        # visual bidirectional rnn encoder
        self.rnn = nn.GRU(opt.visual_feat_dim, opt.visual_rnn_size, batch_first=True, bidirectional=True)

        # visual 1-d convolutional network
        self.convs1 = nn.ModuleList([
            nn.Conv2d(1, opt.visual_kernel_num, (window_size, self.rnn_output_size), padding=(window_size//2, 0))
            for window_size in opt.visual_kernel_sizes
            ])

        self.k1_conv = nn.Conv2d(1, 1, (1, self.rnn_output_size), padding=(0, 0))
        self.k2_conv = nn.Conv2d(1, 1, (1, self.rnn_output_size), padding=(0, 0))
        #self.k3_conv = nn.Conv2d(1, 1, (1, self.rnn_output_size), padding=(0, 0))
        # visual mapping
        self.visual_mapping = MFC(opt.visual_mapping_layers, opt.dropout, have_bn=True, have_last_bn=True)
        self.visual_mapping1 = MFC((2048,2048), opt.dropout, have_bn=True, have_last_bn=True)

    def cal_loss_matrix(self, video, cap_emb):
        # prepare matrix for calculating triplet loss 2_1: (t1,f(t1,v1)) > alpha + (t1,v2(t1))
        bs = video.size(0)
        t = video.size(1)
        video = video.unsqueeze(0)
        video = video.repeat(bs,1,1,1)  # bs,bs,t,2048

        cap_emb = cap_emb.unsqueeze(1)
        cap_emb_bs = cap_emb.repeat(1,bs,1)

        cap_emb = cap_emb.repeat(1, bs, 1)  # bs,bs,2048
        cap_emb = cap_emb.unsqueeze(2) # bs,bs,1,2048

        video_with_text = video * cap_emb
        video_with_text = video_with_text.view(bs*bs,1,t,2048)

        semantic_attention = F.sigmoid(self.k1_conv(video_with_text)) # bs*bs,1,t,1

        video_with_text = video_with_text * semantic_attention
        video_with_text = torch.mean(video_with_text,dim=2,keepdim=False)
        video_with_text = video_with_text.squeeze(1)
        video_with_text = self.visual_mapping1(video_with_text)
        video_with_text = l2norm(video_with_text)
        video_with_text = video_with_text.view(bs,bs,2048)

        sim_matrix = (cap_emb_bs * video_with_text).sum(dim=2,keepdim=False)
        return sim_matrix

    def cal_loss_matrix2(self, video, cap_emb, pure_video_feature):
        # prepare matrix for calculating triplet loss 2_2: (v1,f(t1,v1)) > alpha + (v1,f(v1,t2))
        bs = video.size(0)
        t = video.size(1)
        video = video.unsqueeze(0)
        video = video.repeat(bs,1,1,1)  # bs,bs,t,2048
        pure_video_feature_bs = pure_video_feature.unsqueeze(0)
        pure_video_feature_bs = pure_video_feature_bs.repeat(bs,1,1)


        cap_emb = cap_emb.unsqueeze(1)
        cap_emb_bs = cap_emb.repeat(1,bs,1)

        cap_emb = cap_emb.repeat(1, bs, 1)  # bs,bs,2048
        cap_emb = cap_emb.unsqueeze(2) # bs,bs,1,2048

        video_with_text = video * cap_emb
        video_with_text = video_with_text.view(bs*bs,1,t,2048)

        semantic_attention = F.sigmoid(self.k1_conv(video_with_text)) # bs*bs,1,t,1

        video_with_text = video_with_text * semantic_attention
        video_with_text = torch.mean(video_with_text,dim=2,keepdim=False)
        video_with_text = video_with_text.squeeze(1)
        video_with_text = self.visual_mapping1(video_with_text)
        video_with_text = l2norm(video_with_text)
        video_with_text = video_with_text.view(bs,bs,2048)

        sim_matrix = (pure_video_feature_bs * video_with_text).sum(dim=2,keepdim=False)
        return sim_matrix

    def forward(self, videos, cap_emb, cal_loss_matrix=True):
        """Extract video feature vectors."""


        videos, videos_origin, lengths, vidoes_mask = videos
        # videos shape:  bs,128,2048
        # Level 1. Global Encoding by Mean Pooling According
        org_out = videos_origin


        # Level 2. Temporal-Aware Encoding by biGRU
        gru_init_out, _ = self.rnn(videos)  # bs,128,2048


        mean_gru = Variable(torch.zeros(gru_init_out.size(0), self.rnn_output_size)).cuda()
        for i, batch in enumerate(gru_init_out):
            mean_gru[i] = torch.mean(batch[:lengths[i]], 0)
        gru_out = mean_gru
        gru_out = self.dropout(gru_out)


        # Level 3. Local-Enhanced Encoding by biGRU-CNN
        vidoes_mask = vidoes_mask.unsqueeze(2).expand(-1,-1,gru_init_out.size(2)) # (N,C,F1)
        gru_init_out = gru_init_out * vidoes_mask
        con_out = gru_init_out.unsqueeze(1)
        con_out = [F.relu(conv(con_out)) for conv in self.convs1]
        con_out = torch.cat(con_out,1)  # bs,2048,time_step, 1
        con_out = con_out.permute(0,3,2,1)  # bs,1,time_step, 2048
        self_attention_score = F.sigmoid(self.k2_conv(con_out))

        con_out_for_cal_loss_matrix2 = con_out.squeeze(1)
        cap_emb_for_cal_loss_matrix2 = cap_emb
        ###### Calculate retrieval loss in stage 2 ######
        # loss_matrix = torch.zeros((videos.size(0), videos.size(0))).cuda()
        if cal_loss_matrix:
            loss_matrix = self.cal_loss_matrix(con_out.squeeze(1),cap_emb)
        else:
            loss_matrix = torch.zeros((videos.size(0),videos.size(0))).cuda()


        ###### Calculate response curv ######
        cap_emb = cap_emb.unsqueeze(1)
        cap_emb = cap_emb.unsqueeze(2)
        cap_emb = cap_emb.expand_as(con_out)
        con_out_with_text = con_out * cap_emb
        attention_score = F.sigmoid(self.k1_conv(con_out_with_text)) # bs,1,time_step,1

        con_out_with_text = con_out_with_text * attention_score
        con_out_with_text = torch.mean(con_out_with_text,dim=2,keepdim=False)
        con_out_with_text = con_out_with_text.squeeze(1)

        attention_score = attention_score.squeeze(1)
        attention_score = attention_score.squeeze(2)

        ###### Pure video feature for step 1 retrieval ######
        con_out = con_out * self_attention_score  # bs,1,time_step,2048
        con_out = torch.mean(con_out,dim=2,keepdim=False)
        con_out = con_out.squeeze(1)


        con_out = self.dropout(con_out)

        # concatenation
        if self.concate == 'full': # level 1+2+3
            features = torch.cat((gru_out,con_out,org_out), 1)
            features1 = con_out_with_text
        elif self.concate == 'reduced':  # level 2+3
            features = torch.cat((gru_out,con_out), 1)
            features1 = con_out_with_text


        # mapping to common space
        features = self.visual_mapping(features)
        features1 = self.visual_mapping1(features1)

        if self.visual_norm:
            features = l2norm(features)
            features1 = l2norm(features1)

        if cal_loss_matrix:
            loss_matrix2 = self.cal_loss_matrix2(con_out_for_cal_loss_matrix2,cap_emb_for_cal_loss_matrix2, features)
        else:
            loss_matrix2 = torch.zeros((videos.size(0),videos.size(0))).cuda()

        return features, attention_score, features1, loss_matrix, loss_matrix2

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(Video_multilevel_encoding, self).load_state_dict(new_state)



class Text_multilevel_encoding(nn.Module):
    """
    Section 3.2. Text-side Multi-level Encoding
    """
    def __init__(self, opt):
        super(Text_multilevel_encoding, self).__init__()
        self.text_norm = opt.text_norm
        self.word_dim = opt.word_dim
        self.we_parameter = opt.we_parameter
        self.rnn_output_size = opt.text_rnn_size*2
        self.dropout = nn.Dropout(p=opt.dropout)
        self.concate = opt.concate
        
        # visual bidirectional rnn encoder
        self.embed = nn.Embedding(opt.vocab_size, opt.word_dim)
        self.rnn = nn.GRU(opt.word_dim, opt.text_rnn_size, batch_first=True, bidirectional=True)

        # visual 1-d convolutional network
        self.convs1 = nn.ModuleList([
            nn.Conv2d(1, opt.text_kernel_num, (window_size, self.rnn_output_size), padding=(window_size - 1, 0)) 
            for window_size in opt.text_kernel_sizes
            ])
        
        # multi fc layers
        self.text_mapping = MFC(opt.text_mapping_layers, opt.dropout, have_bn=True, have_last_bn=True)

        self.init_weights()

    def init_weights(self):
        if self.word_dim == 500 and self.we_parameter is not None:
            self.embed.weight.data.copy_(torch.from_numpy(self.we_parameter))
        else:
            self.embed.weight.data.uniform_(-0.1, 0.1)


    def forward(self, text, *args):
        # Embed word ids to vectors
        # cap_wids, cap_w2vs, cap_bows, cap_mask = x
        cap_wids, cap_bows, lengths, cap_mask = text

        # Level 1. Global Encoding by Mean Pooling According
        org_out = cap_bows

        # Level 2. Temporal-Aware Encoding by biGRU
        cap_wids = self.embed(cap_wids)
        packed = pack_padded_sequence(cap_wids, lengths, batch_first=True)
        gru_init_out, _ = self.rnn(packed)
        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(gru_init_out, batch_first=True)
        gru_init_out = padded[0]
        gru_out = Variable(torch.zeros(padded[0].size(0), self.rnn_output_size)).cuda()
        for i, batch in enumerate(padded[0]):
            gru_out[i] = torch.mean(batch[:lengths[i]], 0)
        gru_out = self.dropout(gru_out)

        # Level 3. Local-Enhanced Encoding by biGRU-CNN
        con_out = gru_init_out.unsqueeze(1)
        con_out = [F.relu(conv(con_out)).squeeze(3) for conv in self.convs1]
        con_out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in con_out]
        con_out = torch.cat(con_out, 1)
        con_out = self.dropout(con_out)

        # concatenation
        if self.concate == 'full': # level 1+2+3
            features = torch.cat((gru_out,con_out,org_out), 1)
        elif self.concate == 'reduced': # level 2+3
            features = torch.cat((gru_out,con_out), 1)
        
        # mapping to common space
        features = self.text_mapping(features)
        if self.text_norm:
            features = l2norm(features)

        return features




class BaseModel(object):

    def state_dict(self):
        state_dict = [self.vid_encoding.state_dict(), self.text_encoding.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.vid_encoding.load_state_dict(state_dict[0])
        self.text_encoding.load_state_dict(state_dict[1])

    def train_start(self):
        """switch to train mode
        """
        self.vid_encoding.train()
        self.text_encoding.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.vid_encoding.eval()
        self.text_encoding.eval()


    def forward_loss(self, cap_emb, vid_emb, atten_score, atten_score_gt, loss_matrix, loss_matrix2, *agrs, **kwargs):
        """Compute the loss given pairs of video and caption embeddings
        """
        loss = self.criterion(cap_emb, vid_emb, atten_score, atten_score_gt, loss_matrix, loss_matrix2)
        if torch.__version__ == '0.3.1':  # loss.item() for 0.4.0, loss.data[0] for 0.3.1
            self.logger.update('Le', loss.data[0], vid_emb.size(0)) 
        else:
            self.logger.update('Le', loss.item(), vid_emb.size(0)) 
        return loss

    def train_emb(self, videos, captions, lengths, *args):
        """One training step given videos and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        cap_ids, video_ids, seg_ids, timestamps, durations, atten_score_gt = args

        # compute the embeddings
        vid_emb, cap_emb, atten_score, vid_emb_with_text, loss_matrix, loss_matrix2 = self.forward_emb(videos, captions, False)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(cap_emb, vid_emb, atten_score, atten_score_gt, loss_matrix, loss_matrix2)
        
        if torch.__version__ == '0.3.1':
            loss_value = loss.data[0]
        else:
            loss_value = loss.item()

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()

        return cap_emb.size(0), loss_value



class Dual_Encoding(BaseModel):
    """
    dual encoding network
    """
    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip
        self.vid_encoding = Video_multilevel_encoding(opt)
        self.text_encoding = Text_multilevel_encoding(opt)
        print(self.vid_encoding)
        print(self.text_encoding)
        if torch.cuda.is_available():
            self.vid_encoding.cuda()
            self.text_encoding.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        if opt.loss_fun == 'mrl':
            self.criterion = TripletLoss(margin=opt.margin,
                                            measure=opt.measure,
                                            max_violation=opt.max_violation,
                                            cost_style=opt.cost_style,
                                            direction=opt.direction)

        params = list(self.text_encoding.parameters())
        params += list(self.vid_encoding.parameters())
        self.params = params

        if opt.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)
        elif opt.optimizer == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(params, lr=opt.learning_rate)

        self.Eiters = 0


    def forward_emb(self, videos, targets, volatile=False, repeat=0, *args):
        """Compute the video and caption embeddings
        """
        # video data
        frames, mean_origin, video_lengths, vidoes_mask = videos
        if volatile:
            with torch.no_grad():
                frames = Variable(frames)
                if torch.cuda.is_available():
                    frames = frames.cuda()

                mean_origin = Variable(mean_origin)
                if torch.cuda.is_available():
                    mean_origin = mean_origin.cuda()

                vidoes_mask = Variable(vidoes_mask)
                if torch.cuda.is_available():
                    vidoes_mask = vidoes_mask.cuda()
                videos_data = (frames, mean_origin, video_lengths, vidoes_mask)

                # text data
                captions, cap_bows, lengths, cap_masks = targets
                if captions is not None:
                    captions = Variable(captions)
                    if torch.cuda.is_available():
                        captions = captions.cuda()

                if cap_bows is not None:
                    cap_bows = Variable(cap_bows)
                    if torch.cuda.is_available():
                        cap_bows = cap_bows.cuda()

                if cap_masks is not None:
                    cap_masks = Variable(cap_masks)
                    if torch.cuda.is_available():
                        cap_masks = cap_masks.cuda()
                text_data = (captions, cap_bows, lengths, cap_masks)
        else:
            frames = Variable(frames)
            if torch.cuda.is_available():
                frames = frames.cuda()

            mean_origin = Variable(mean_origin)
            if torch.cuda.is_available():
                mean_origin = mean_origin.cuda()

            vidoes_mask = Variable(vidoes_mask)
            if torch.cuda.is_available():
                vidoes_mask = vidoes_mask.cuda()
            videos_data = (frames, mean_origin, video_lengths, vidoes_mask)

            # text data
            captions, cap_bows, lengths, cap_masks = targets
            if captions is not None:
                captions = Variable(captions)
                if torch.cuda.is_available():
                    captions = captions.cuda()

            if cap_bows is not None:
                cap_bows = Variable(cap_bows)
                if torch.cuda.is_available():
                    cap_bows = cap_bows.cuda()

            if cap_masks is not None:
                cap_masks = Variable(cap_masks)
                if torch.cuda.is_available():
                    cap_masks = cap_masks.cuda()
            text_data = (captions, cap_bows, lengths, cap_masks)



        cap_emb = self.text_encoding(text_data)

        if repeat:
            cap_emb = cap_emb.repeat(repeat,1)


        vid_emb, atten_score, vid_emb_with_text, loss_matrix, loss_matrix2 = self.vid_encoding(videos_data, cap_emb, cal_loss_matrix = not volatile)
        
        return vid_emb, cap_emb, atten_score, vid_emb_with_text, loss_matrix, loss_matrix2



NAME_TO_MODELS = {'dual_encoding': Dual_Encoding}

def get_model(name):
    assert name in NAME_TO_MODELS, '%s not supported.'%name
    return NAME_TO_MODELS[name]
