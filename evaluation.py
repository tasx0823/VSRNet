from __future__ import print_function
import os
import pickle

import numpy
import time
import numpy as np
import torch
from torch.autograd import Variable
from basic.metric import getScorer
from basic.util import AverageMeter, LogCollector


def l2norm(X):
	"""L2-normalize columns of X
	"""
	norm = np.linalg.norm(X, axis=1, keepdims=True)
	return 1.0 * X / norm


def cal_error(videos, captions, measure='cosine'):
	videos = np.squeeze(videos)
	if measure == 'cosine':
		captions = l2norm(captions)
		videos = l2norm(videos)
		errors = -1 * numpy.dot(captions, videos.T)

	return errors


def cal_error_gpu(videos, captions, measure='cosine'):
	def l2norm_gpu(x):
		norm = torch.norm(x, p=2, dim=1, keepdim=True)
		return 1.0 * x / norm

	captions = l2norm_gpu(captions)
	videos = l2norm_gpu(videos)
	errors = -1 * torch.dot(captions, videos.T)

	return errors


def cal_error__(videos, captions, measure='cosine'):
	if measure == 'cosine':
		# captions = l2norm(captions)
		# videos = l2norm(videos)
		# errors = -1*numpy.dot(captions, videos.T)
		captions = l2norm(captions)
		videos = l2norm(videos)
		errors = numpy.dot(videos, captions.T)
		errors = np.reshape(errors, (-1, 7, np.shape(captions)[0]))
		errors = np.max(errors, axis=1, keepdims=False)
		errors = -1 * errors.T

	return errors


def cal_error_(videos, captions, measure='cosine'):
	if measure == 'cosine':
		# captions = l2norm(captions)
		# videos = l2norm(videos)
		# errors = -1*numpy.dot(captions, videos.T)
		captions = l2norm(captions)
		videos = l2norm(videos)
		errors = numpy.dot(videos, captions.T)
		errors = np.reshape(errors, (-1, 7, np.shape(captions)[0]))
		max_idxs = np.argmax(errors, axis=1)
		errors = np.max(errors, axis=1, keepdims=False)
		errors = -1 * errors.T

	return errors, max_idxs.T


def encode_data(model, data_loader, log_step=10, logging=print, return_ids=True):
	"""Encode all videos and captions loadable by `data_loader`
	"""
	batch_time = AverageMeter()
	val_logger = LogCollector()

	# switch to evaluate mode
	model.val_start()

	end = time.time()

	# numpy array to keep all the embeddings
	video_embs = None
	cap_embs = None
	atten_scores = None
	video_ids = [''] * len(data_loader.dataset)
	caption_ids = [''] * len(data_loader.dataset)
	for i, (videos, captions, idxs, cap_ids, vid_ids, seg_ids, timestamp, duration, gt_attention_scores) in enumerate(
			data_loader):
		# make sure val logger is used
		model.logger = val_logger

		# compute the embeddings
		vid_emb, cap_emb, atten_score, vid_emb_with_text, loss_matrix, loss_matrix2 = model.forward_emb(videos,
		                                                                                                captions, True)

		# initialize the numpy arrays given the size of the embeddings
		if video_embs is None:
			video_embs = np.zeros((len(data_loader.dataset), vid_emb.size(1)))
			atten_scores = np.zeros((len(data_loader.dataset), 128))

		if cap_embs is None:
			cap_embs = np.zeros((len(data_loader.dataset), cap_emb.size(1)))
			timestamps = np.zeros((len(data_loader.dataset), 2))
			durations = np.zeros((len(data_loader.dataset)))

		video_embs[idxs] = vid_emb.data.cpu().numpy().copy()
		cap_embs[idxs] = cap_emb.data.cpu().numpy().copy()
		atten_scores[idxs] = atten_score.data.cpu().numpy().copy()
		timestamps[idxs] = timestamp
		durations[idxs] = duration
		for j, idx in enumerate(idxs):
			caption_ids[idx] = cap_ids[j]
			video_ids[idx] = vid_ids[j]

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % log_step == 0:
			logging('Test: [{0:2d}/{1:2d}]\t'
			        '{e_log}\t'
			        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
			        .format(i, len(data_loader), batch_time=batch_time,
			                e_log=str(model.logger)))

		del videos, captions

	# video_embs = np.reshape(video_embs,(-1,2048))
	if return_ids == True:
		return video_embs, cap_embs, video_ids, caption_ids, timestamps, durations, atten_scores
	else:
		return video_embs, cap_embs


def cal_iou_using_id(pre_seg_id, timestamp, duration):
	segs = [[0, 1], [0, 0.5], [0.5, 1], [0, 0.25], [0.25, 0.5], [0.5, 0.75], [0.75, 1]]
	seg = segs[pre_seg_id]
	timestamp = 1.0 * np.array(timestamp) / duration

	union = (min(seg[0], timestamp[0]), max(seg[1], timestamp[1]))
	inter = (max(seg[0], timestamp[0]), min(seg[1], timestamp[1]))
	iou = 1.0 * (inter[1] - inter[0]) / (union[1] - union[0])
	return iou


def cal_iou(pre_time, gt_time):
	union = (min(pre_time[0], gt_time[0]), max(pre_time[1], gt_time[1]))
	inter = (max(pre_time[0], gt_time[0]), min(pre_time[1], gt_time[1]))
	iou = 1.0 * (inter[1] - inter[0]) / (union[1] - union[0])
	return iou


def cal_grounding_accuracy_0_dot_7(file):
	f1 = open(file, 'r')
	lines = f1.readlines()
	iou_cnt = 0.0
	cnt = 0.0
	iou_sum = 0.0

	for line in lines:
		line = line.strip()
		line = line.split(' ', 3)
		# video_id = line[0]
		start = float(line[0])
		end = float(line[1])
		duration = float(line[2])
		attn_scores = line[3].split(' ')
		attn_scores = np.array(attn_scores).astype(np.float)

		max_score = np.max(attn_scores)
		th = 0.33 * max_score
		# th = 0.3
		pre_s = 0
		while (pre_s < 127 and attn_scores[pre_s] < th):
			pre_s += 1
		pre_e = 127
		while (pre_e > pre_s and attn_scores[pre_e] < th):
			pre_e -= 1
		pre_time = 1.0 * np.array((pre_s, pre_e)) / 127
		gt_time = 1.0 * np.array((start, end)) / duration

		# print(gt_time)

		iou = cal_iou(pre_time, gt_time)
		if iou < 0:
			iou = 0
		if iou >= 0.7:
			iou_cnt += 1
		cnt += 1
		iou_sum += iou

	return iou_cnt / cnt


def cal_grounding_accuracy(file):
	f1 = open(file, 'r')
	lines = f1.readlines()
	iou_cnt = 0.0
	cnt = 0.0
	iou_sum = 0.0

	for line in lines:
		line = line.strip()
		line = line.split(' ', 3)
		# video_id = line[0]
		start = float(line[0])
		end = float(line[1])
		duration = float(line[2])
		attn_scores = line[3].split(' ')
		attn_scores = np.array(attn_scores).astype(np.float)

		max_score = np.max(attn_scores)
		th = 0.33 * max_score
		# th = 0.3
		pre_s = 0
		while (pre_s < 127 and attn_scores[pre_s] < th):
			pre_s += 1
		pre_e = 127
		while (pre_e > pre_s and attn_scores[pre_e] < th):
			pre_e -= 1
		pre_time = 1.0 * np.array((pre_s, pre_e)) / 127
		gt_time = 1.0 * np.array((start, end)) / duration

		# print(gt_time)

		iou = cal_iou(pre_time, gt_time)
		if iou < 0:
			iou = 0
		if iou >= 0.5:
			iou_cnt += 1
		cnt += 1
		iou_sum += iou

	return iou_cnt / cnt


def is_retrieved_segment_correct(start, end, duration, attn_scores):
	# For temporal localization task, we set the threshold gamma to 0.3
	gamma = 0.33
	pre_s = 0
	max_score = np.max(attn_scores)
	th = max_score * gamma
	while (pre_s < 127 and attn_scores[pre_s] < th):
		pre_s += 1
	pre_e = 127
	while (pre_e > pre_s and attn_scores[pre_e] < th):
		pre_e -= 1
	pre_time = 1.0 * np.array((pre_s, pre_e)) / 127
	gt_time = 1.0 * np.array((start, end)) / duration
	iou = cal_iou(pre_time, gt_time)
	if iou > 0.5:
		return True
	return False

def is_retrieved_segment_correct_0_7(start, end, duration, attn_scores):
	th = 0.33
	pre_s = 0
	while (pre_s < 127 and attn_scores[pre_s] < th):
		pre_s += 1
	pre_e = 127
	while (pre_e > pre_s and attn_scores[pre_e] < th):
		pre_e -= 1
	pre_time = 1.0 * np.array((pre_s, pre_e)) / 127
	gt_time = 1.0 * np.array((start, end)) / duration
	iou = cal_iou(pre_time, gt_time)
	if iou > 0.7:
		return True
	return False

# recall@k, Med r, Mean r for Text-to-Video Retrieval
def t2i(c2i, timestamps, durations, atten_scores, opt, vis_details=False, n_caption=5, topK=30, ):
	"""
	Text->Videos (Text-to-Video Retrieval)
	c2i: (5N, N) matrix of caption to video errors
	vis_details: if true, return a dictionary for ROC visualization purposes
	"""
	# print("errors matrix shape: ", c2i.shape)
	assert c2i.shape[0] / c2i.shape[1] == n_caption, c2i.shape
	assert c2i.shape[0] == timestamps.shape[0]
	ranks = np.zeros(c2i.shape[0])
	top1_cnt = 0
	iou_sum = 0.0
	iou_cnt_0_dot_3 = 0
	iou_cnt_0_dot_5 = 0
	iou_cnt_0_dot_7 = 0
	iou_cnt_0_dot_9 = 0
	topK_candidates = np.zeros((c2i.shape[0], topK)).astype(np.int)
	f1 = open(os.path.join(opt.logger_name, 'rank1_s_t_and_duration1.txt'), 'w')
	f10 = open(os.path.join(opt.logger_name, 'rank1_s_t_and_duration10.txt'), 'w')
	f100 = open(os.path.join(opt.logger_name, 'rank1_s_t_and_duration100.txt'), 'w')
	f_attention_scores = open(os.path.join(opt.logger_name, 'attention_scores.txt'), 'w')

	acc_at_0_dot_5_1 = 0
	acc_at_0_dot_5_10 = 0
	acc_at_0_dot_5_100 = 0
	acc_at_0_dot_5_1_total = 0
	acc_at_0_dot_5_10_total = 0
	acc_at_0_dot_5_100_total = 0

	acc_0_5_fenzi = 0.0
	acc_0_7_fenzi = 0.0
	for i in range(len(ranks)):

		# -----------------------------------------------------------------------------#
		# Calculate acc@0.5 for every video
		if is_retrieved_segment_correct(timestamps[i][0], timestamps[i][1], durations[i], atten_scores[i]):
			acc_0_5_fenzi += 1
		if is_retrieved_segment_correct_0_7(timestamps[i][0], timestamps[i][1], durations[i], atten_scores[i]):
			acc_0_7_fenzi += 1
		# -----------------------------------------------------------------------------#
		d_i = c2i[i]
		inds = np.argsort(d_i)
		topK_candidates[i] = inds[:topK]
		rank = np.where(inds == i / n_caption)[0][0]

		f_attention_scores.write('{:f} {:f} {:f} '.format(timestamps[i][0], timestamps[i][1], durations[i]))
		f_attention_scores.write(str(list(atten_scores[i])).replace('[', '').replace(']', '').replace(',', ''))
		f_attention_scores.write('\n')

		ranks[i] = rank
		if (rank == 0):
			top1_cnt += 1
			# pre_seg_id = max_idx[i/n_caption]
			# pre_seg_id = np.random.randint(7)
			pre_seg_id = 0
			iou = cal_iou_using_id(pre_seg_id, timestamps[i], durations[i])
			f1.write('{:f} {:f} {:f} '.format(timestamps[i][0], timestamps[i][1], durations[i]))
			f1.write(str(list(atten_scores[i])).replace('[', '').replace(']', '').replace(',', ''))
			f1.write('\n')
			if is_retrieved_segment_correct(timestamps[i][0], timestamps[i][1], durations[i], atten_scores[i]):
				acc_at_0_dot_5_1 += 1
			acc_at_0_dot_5_1_total += 1
			iou_sum += iou
			if iou >= 0.3:
				iou_cnt_0_dot_3 += 1
			if iou >= 0.5:
				iou_cnt_0_dot_5 += 1
			if iou >= 0.7:
				iou_cnt_0_dot_7 += 1
			if iou >= 0.9:
				iou_cnt_0_dot_9 += 1

		if (rank < 10):
			f10.write('{:f} {:f} {:f} '.format(timestamps[i][0], timestamps[i][1], durations[i]))
			f10.write(str(list(atten_scores[i])).replace('[', '').replace(']', '').replace(',', ''))
			f10.write('\n')
			if is_retrieved_segment_correct(timestamps[i][0], timestamps[i][1], durations[i], atten_scores[i]):
				acc_at_0_dot_5_10 += 1
			acc_at_0_dot_5_10_total += 1

		if (rank < 100):
			f100.write('{:f} {:f} {:f} '.format(timestamps[i][0], timestamps[i][1], durations[i]))
			f100.write(str(list(atten_scores[i])).replace('[', '').replace(']', '').replace(',', ''))
			f100.write('\n')
			if is_retrieved_segment_correct(timestamps[i][0], timestamps[i][1], durations[i], atten_scores[i]):
				acc_at_0_dot_5_100 += 1
			acc_at_0_dot_5_100_total += 1

	f1.close()
	f10.close()
	f100.close()
	f_attention_scores.close()

	acc_at_0_dot_5_1 = cal_grounding_accuracy(os.path.join(opt.logger_name, 'rank1_s_t_and_duration1.txt'))
	acc_at_0_dot_5_10 = cal_grounding_accuracy(os.path.join(opt.logger_name, 'rank1_s_t_and_duration10.txt'))
	acc_at_0_dot_5_100 = cal_grounding_accuracy(os.path.join(opt.logger_name, 'rank1_s_t_and_duration100.txt'))

	acc_at_0_dot_7_1 = cal_grounding_accuracy_0_dot_7(os.path.join(opt.logger_name, 'rank1_s_t_and_duration1.txt'))
	acc_at_0_dot_7_10 = cal_grounding_accuracy_0_dot_7(os.path.join(opt.logger_name, 'rank1_s_t_and_duration10.txt'))
	acc_at_0_dot_7_100 = cal_grounding_accuracy_0_dot_7(os.path.join(opt.logger_name, 'rank1_s_t_and_duration100.txt'))

	print(os.path.join(opt.logger_name, 'rank1_s_t_and_duration1.txt'))

	# acc_at_0_dot_5_1 = 100.0 * acc_at_0_dot_5_1 / acc_at_0_dot_5_1_total
	# acc_at_0_dot_5_10 = 100.0 * acc_at_0_dot_5_10 / acc_at_0_dot_5_10_total
	# acc_at_0_dot_5_100 = 100.0 * acc_at_0_dot_5_100 / acc_at_0_dot_5_100_total

	acc_at_0_dot_5_7 = [acc_at_0_dot_5_1, acc_at_0_dot_5_10, acc_at_0_dot_5_100, acc_at_0_dot_7_1, acc_at_0_dot_7_10,
	                    acc_at_0_dot_7_100]

	mean_iou = iou_sum / top1_cnt
	acc_0_dot_3 = 1.0 * iou_cnt_0_dot_3 / top1_cnt
	acc_0_dot_5 = 1.0 * iou_cnt_0_dot_5 / top1_cnt
	acc_0_dot_7 = 1.0 * iou_cnt_0_dot_7 / top1_cnt
	acc_0_dot_9 = 1.0 * iou_cnt_0_dot_9 / top1_cnt

	# Compute metrics
	r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
	r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
	r100 = 100.0 * len(np.where(ranks < 100)[0]) / len(ranks)
	medr = np.floor(np.median(ranks)) + 1
	meanr = ranks.mean() + 1

	

	return map(float, [r1, r10, r100, medr, meanr]), mean_iou, [acc_0_dot_3, acc_0_dot_5, acc_0_dot_7,
	                                                            acc_0_dot_9], topK_candidates, acc_at_0_dot_5_7


# recall@k, Med r, Mean r for Text-to-Video Retrieval
def t2i_(c2i, vis_details=False, n_caption=5):
	"""
	Text->Videos (Text-to-Video Retrieval)
	c2i: (5N, N) matrix of caption to video errors
	vis_details: if true, return a dictionary for ROC visualization purposes
	"""
	# print("errors matrix shape: ", c2i.shape)
	assert c2i.shape[0] / c2i.shape[1] == n_caption, c2i.shape
	ranks = np.zeros(c2i.shape[0])

	for i in range(len(ranks)):
		d_i = c2i[i]
		inds = np.argsort(d_i)

		rank = np.where(inds == i / n_caption)[0][0]
		ranks[i] = rank

	# Compute metrics
	r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
	r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
	r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
	medr = np.floor(np.median(ranks)) + 1
	meanr = ranks.mean() + 1

	return map(float, [r1, r5, r10, medr, meanr])


# recall@k, Med r, Mean r for Video-to-Text Retrieval
def i2t(c2i, n_caption=5):
	"""
	Videos->Text (Video-to-Text Retrieval)
	c2i: (5N, N) matrix of caption to video errors
	"""
	# remove duplicate videos
	# print("errors matrix shape: ", c2i.shape)
	assert c2i.shape[0] / c2i.shape[1] == n_caption, c2i.shape
	ranks = np.zeros(c2i.shape[1])

	for i in range(len(ranks)):
		d_i = c2i[:, i]
		inds = np.argsort(d_i)

		rank = np.where(inds / n_caption == i)[0][0]
		ranks[i] = rank

	# Compute metrics
	r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
	r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
	r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
	medr = np.floor(np.median(ranks)) + 1
	meanr = ranks.mean() + 1
	return map(float, [r1, r5, r10, medr, meanr])


# mAP for Text-to-Video Retrieval
def t2i_map(c2i, n_caption=5):
	"""
	Text->Videos (Text-to-Video Retrieval)
	c2i: (5N, N) matrix of caption to video errors
	"""
	# print("errors matrix shape: ", c2i.shape)
	assert c2i.shape[0] / c2i.shape[1] == n_caption, c2i.shape

	scorer = getScorer('AP')
	perf_list = []
	for i in range(c2i.shape[0]):
		d_i = c2i[i, :]
		labels = [0] * len(d_i)
		labels[i / n_caption] = 1

		sorted_labels = [labels[x] for x in np.argsort(d_i)]
		current_score = scorer.score(sorted_labels)
		perf_list.append(current_score)

	return np.mean(perf_list)


# mAP for Video-to-Text Retrieval
def i2t_map(c2i, n_caption=5):
	"""
	Videos->Text (Video-to-Text Retrieval)
	c2i: (5N, N) matrix of caption to video errors
	"""
	# print("errors matrix shape: ", c2i.shape)
	assert c2i.shape[0] / c2i.shape[1] == n_caption, c2i.shape

	scorer = getScorer('AP')
	perf_list = []
	for i in range(c2i.shape[1]):
		d_i = c2i[:, i]
		labels = [0] * len(d_i)
		labels[i * n_caption:(i + 1) * n_caption] = [1] * n_caption

		sorted_labels = [labels[x] for x in np.argsort(d_i)]
		current_score = scorer.score(sorted_labels)
		perf_list.append(current_score)

	return np.mean(perf_list)


def t2i_inv_rank(c2i, n_caption=1):
	"""
	Text->Videos (Text-to-Video Retrieval)
	c2i: (5N, N) matrix of caption to video errors
	n_caption: number of captions of each image/video
	"""
	assert c2i.shape[0] / c2i.shape[1] == n_caption, c2i.shape
	inv_ranks = np.zeros(c2i.shape[0])

	for i in range(len(inv_ranks)):
		d_i = c2i[i, :]
		inds = np.argsort(d_i)

		rank = np.where(inds == i / n_caption)[0]
		inv_ranks[i] = sum(1.0 / (rank + 1))

	return np.mean(inv_ranks)


def i2t_inv_rank(c2i, n_caption=1):
	"""
	Videos->Text (Video-to-Text Retrieval)
	c2i: (5N, N) matrix of caption to video errors
	n_caption: number of captions of each image/video
	"""
	assert c2i.shape[0] / c2i.shape[1] == n_caption, c2i.shape
	inv_ranks = np.zeros(c2i.shape[1])

	for i in range(len(inv_ranks)):
		d_i = c2i[:, i]
		inds = np.argsort(d_i)

		rank = np.where(inds / n_caption == i)[0]
		inv_ranks[i] = sum(1.0 / (rank + 1))

	return np.mean(inv_ranks)


def i2t_inv_rank_multi(c2i, n_caption=2):
	"""
	Text->videos (Image Search)
	c2i: (5N, N) matrix of caption to image errors
	n_caption: number of captions of each image/video
	"""
	# print("errors matrix shape: ", c2i.shape)
	assert c2i.shape[0] / c2i.shape[1] == n_caption, c2i.shape
	inv_ranks = np.zeros(c2i.shape[1])

	result = []
	for i in range(n_caption):
		idx = range(i, c2i.shape[0], n_caption)
		sub_c2i = c2i[idx, :]
		score = i2t_inv_rank(sub_c2i, n_caption=1)
		result.append(score)
	return result
