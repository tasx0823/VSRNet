from __future__ import print_function
import pickle
import os
import sys

import torch

import evaluation
from model import get_model
import util.data_provider as data
from util.vocab import Vocabulary
from util.text2vec import get_text_encoder

import logging
import json
import numpy as np

import argparse
from basic.util import read_dict
from basic.constant import ROOT_PATH
from basic.bigfile import BigFile
from basic.common import makedirsforfile, checkToSkip

def parse_args():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--testCollection', type=str,default='densecapval', help='test collection')
    parser.add_argument('--rootpath', type=str, default=ROOT_PATH, help='path to datasets. (default: %s)'%ROOT_PATH)
    parser.add_argument('--overwrite', type=int, default=0, choices=[0,1],  help='overwrite existed file. (default: 0)')
    parser.add_argument('--log_step', default=10, type=int, help='Number of steps to print and record the log.')
    parser.add_argument('--batch_size', default=96, type=int, help='Size of a training mini-batch.')
    parser.add_argument('--workers', default=5, type=int, help='Number of data loader workers.')
    parser.add_argument('--logger_name', default='ckpt/jiaojie', help='Path to save the model and Tensorboard log.')
    parser.add_argument('--checkpoint_name', default='model_best.pth.tar', type=str, help='name of checkpoint (default: model_best.pth.tar)')
    parser.add_argument('--n_caption', type=int, default=5, help='number of captions of each image/video (default: 1)')
    parser.add_argument('--gpu_no',type=str,default='0',help='assign which gpu to use')
    args = parser.parse_args()
    return args

def load_config(config_path):
    variables = {}
    exec(compile(open(config_path, "rb").read(), config_path, 'exec'), variables)
    return variables['config']

def main():
    opt = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_no
    print(json.dumps(vars(opt), indent=2))

    rootpath = opt.rootpath
    testCollection = opt.testCollection
    n_caption = opt.n_caption
    resume = os.path.join(opt.logger_name, opt.checkpoint_name)

    if not os.path.exists(resume):
        logging.info(resume + ' not exists.')
        sys.exit(0)


    checkpoint = torch.load(resume)
    start_epoch = checkpoint['epoch']
    best_rsum = checkpoint['best_rsum']
    print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
          .format(resume, start_epoch, best_rsum))
    options = checkpoint['opt']
    if not hasattr(options, 'concate'):
        setattr(options, "concate", "full")

    trainCollection = options.trainCollection
    output_dir = resume.replace(trainCollection, testCollection)
    output_dir = output_dir.replace('/%s/' % options.cv_name, '/results/%s/' % trainCollection )
    result_pred_sents = os.path.join(output_dir, 'id.sent.score.txt')
    pred_error_matrix_file = os.path.join(output_dir, 'pred_errors_matrix.pth.tar')
    if checkToSkip(pred_error_matrix_file, opt.overwrite):
        sys.exit(0)
    makedirsforfile(pred_error_matrix_file)

    # data loader prepare
    caption_files = {'test': os.path.join(rootpath, testCollection, 'TextData', '%s.caption.txt'%testCollection)}
    img_feat_path = os.path.join(rootpath, testCollection, 'FeatureData', options.visual_feature)
    visual_feats = {'test': BigFile(img_feat_path)}
    assert options.visual_feat_dim == visual_feats['test'].ndims
    video2frames = {'test': read_dict(os.path.join(rootpath, testCollection, 'FeatureData', options.visual_feature, 'video2frames.txt'))}

    # set bow vocabulary and encoding
    bow_vocab_file = os.path.join(rootpath, options.trainCollection, 'TextData', 'vocabulary', 'bow', options.vocab+'.pkl')
    bow_vocab = pickle.load(open(bow_vocab_file, 'rb'))
    bow2vec = get_text_encoder('bow')(bow_vocab)
    options.bow_vocab_size = len(bow_vocab)

    # set rnn vocabulary 
    rnn_vocab_file = os.path.join(rootpath, options.trainCollection, 'TextData', 'vocabulary', 'rnn', options.vocab+'.pkl')
    rnn_vocab = pickle.load(open(rnn_vocab_file, 'rb'))
    options.vocab_size = len(rnn_vocab)

    # set data loader
    data_loader = data.get_test_data_loaders(
        caption_files, visual_feats, rnn_vocab, bow2vec, opt.batch_size, opt.workers, opt.n_caption, video2frames=video2frames)

    # Construct the model
    model = get_model(options.model)(options)
    model.load_state_dict(checkpoint['model'])
    model.Eiters = checkpoint['Eiters']

    video_embs, cap_embs, video_ids, caption_ids, timestamps, durations, atten_scores = evaluation.encode_data(model, data_loader['test'], opt.log_step, logging.info)

    # remove duplicate videos
    #idx = range(0, video_embs.shape[0], n_caption)
    #video_embs = video_embs[idx,:]

    # -------------------------------------- #
    # we load data as video-sentence pairs
    # but we only need to forward each video once for evaluation
    # so we get the video set and mask out same videos with feature_mask
    feature_mask = []
    evaluate_videos = set()
    for video_id in video_ids:
        feature_mask.append(video_id not in evaluate_videos)
        evaluate_videos.add(video_id)
    video_embs = video_embs[feature_mask]
    video_ids = [x for idx, x in enumerate(video_ids) if feature_mask[idx] is True]

    # print('video_embeds shape ',np.shape(video_embs))
    # print('cap_embs shape ',np.shape(cap_embs))

    video_embs = np.reshape(video_embs, (-1, 2048))
    # -------------------------------------- #
    c2i_all_errors = evaluation.cal_error(video_embs, cap_embs, options.measure)

    txt_path = 'jiaojie_log.txt'
    with open(txt_path, 'a+') as f:
        # video retrieval
        (r1i, r5i, r10i, medri, meanri), mean_iou, acc, topK_candidates, acc_at_0_dot_5 = evaluation.t2i(c2i_all_errors,
                                                                                                         timestamps,
                                                                                                         durations,
                                                                                                         atten_scores,
                                                                                                         opt,
                                                                                                         n_caption=opt.n_caption,
                                                                                                         topK=30)

        video_retrieval_performance = np.array([r1i, r5i, r10i])
        print(" * Text to video:")
        print(" * r_1_5_10: {}".format([round(r1i, 3), round(r5i, 3), round(r10i, 3)]))
        print(" * medr, meanr: {}".format([round(medri, 3), round(meanri, 3)]))
        print(" * acc@0.5: {} || {}".format(acc_at_0_dot_5[:3], video_retrieval_performance * acc_at_0_dot_5[:3]))
        print(" * acc@0.7: {} || {}".format(acc_at_0_dot_5[3:], video_retrieval_performance * acc_at_0_dot_5[3:]))

        print(" * " + '-' * 10)

        f.write("* Text to video:\n")
        f.write(" * r_1_5_10: {}\n".format([round(r1i, 3), round(r5i, 3), round(r10i, 3)]))
        f.write(" * medr, meanr: {}\n".format([round(medri, 3), round(meanri, 3)]))
        f.write(" * acc@0.5: {}\n".format(acc_at_0_dot_5[:3]))
        f.write(" * acc@0.7: {}\n".format(acc_at_0_dot_5[3:]))
        f.write(" * " + '-' * 10 + '\n')


if __name__ == '__main__':
    main()
