import torch
import torch.utils.data as data
import numpy as np
import json as jsonmod

from basic.util import getVideoId
from vocab import clean_str
import random
VIDEO_MAX_LEN = 128
import cv2

def collate_frame_gru_fn(data):
	"""
	Build mini-batch tensors from a list of (video, caption) tuples.
	"""
	# Sort a data list by caption length
	if data[0][1] is not None:
		data.sort(key=lambda x: len(x[1]), reverse=True)
	videos, captions, cap_bows, idxs, cap_ids, video_ids, seg_ids, timestamps, durations, gt_attention_scores = zip(*data)

	# Merge videos (convert tuple of 1D tensor to 4D tensor)
	video_lengths = [min(VIDEO_MAX_LEN, len(frame)) for frame in videos]
	frame_vec_len = len(videos[0][0])
	vidoes = torch.zeros(len(videos), max(video_lengths), frame_vec_len)
	videos_origin = torch.zeros(len(videos), frame_vec_len)
	vidoes_mask = torch.zeros(len(videos), max(video_lengths))
	for i, frames in enumerate(videos):
		end = video_lengths[i]
		vidoes[i, :end, :] = frames[:end, :]
		videos_origin[i, :] = torch.mean(frames, 0)
		vidoes_mask[i, :end] = 1.0

	if captions[0] is not None:
		# Merge captions (convert tuple of 1D tensor to 2D tensor)
		lengths = [len(cap) for cap in captions]
		target = torch.zeros(len(captions), max(lengths)).long()
		words_mask = torch.zeros(len(captions), max(lengths))
		for i, cap in enumerate(captions):
			end = lengths[i]
			target[i, :end] = cap[:end]
			words_mask[i, :end] = 1.0
	else:
		target = None
		lengths = None
		words_mask = None

	gt_attention_scores_ret = torch.zeros(len(gt_attention_scores),128)
	for i in range(len(gt_attention_scores)):
		gt_attention_scores_ret[i] = gt_attention_scores[i]

	cap_bows = torch.stack(cap_bows, 0) if cap_bows[0] is not None else None

	video_data = (vidoes, videos_origin, video_lengths, vidoes_mask)
	text_data = (target, cap_bows, lengths, words_mask)



	return video_data, text_data, idxs, cap_ids, video_ids, seg_ids, timestamps, durations, gt_attention_scores_ret


class Dataset4DualEncoding(data.Dataset):
	"""
	Load captions and video frame features by pre-trained CNN model.
	"""

	def __init__(self, cap_file, visual_feat, bow2vec, vocab, n_caption=None, video2frames=None):
		# Captions
		self.captions = {}
		self.cap_ids = []
		self.seg_ids = []
		self.timestamps = []
		self.durations = []
		self.video_ids = set()
		self.video2frames = video2frames
		with open(cap_file, 'r') as cap_reader:
			print('cap file:', cap_file)
			for line in cap_reader.readlines():
				# try:
				cap_id, seg_id, start_time, end_time, duration, caption = line.strip().split(' ', 5)
				video_id = getVideoId(cap_id)
				video_id = video_id.replace('\xef\xbb\xbf','')
				self.captions[cap_id] = caption
				self.cap_ids.append(cap_id)
				self.seg_ids.append(int(seg_id))
				self.video_ids.add(video_id)
				self.timestamps.append((start_time,end_time))
				self.durations.append(duration)



		# except:
		#	print(line)

		# check

		self.visual_feat = visual_feat
		self.bow2vec = bow2vec
		self.vocab = vocab
		self.length = len(self.cap_ids)

		if n_caption is not None:
			print("n caption is ",n_caption)
			assert len(self.video_ids) * n_caption == self.length, "%d != %d" % (
				len(self.video_ids) * n_caption, self.length)

	def __getitem__(self, index):
		cap_id = self.cap_ids[index]
		seg_id = self.seg_ids[index]
		timestamp = self.timestamps[index]
		duration = self.durations[index]

		start_idx = int(1.0 * float(timestamp[0]) / float(duration) * 128)
		end_idx = int(1.0 * float(timestamp[1]) / float(duration) * 128)
		gt_attention_score = np.zeros(128)
		gt_attention_score[start_idx:end_idx] = 1

		#print('cap_id  ',cap_id)
		video_id = getVideoId(cap_id)

		# video
		#print(video_id)
		video_id = video_id.replace('\xef\xbb\xbf','').strip()
		full_frame_list = self.video2frames[video_id]
		frame_list = full_frame_list
		#idxs = random.sample(range(len(full_frame_list)),8)
		#idxs = sorted(idxs)

		#for idx in idxs:
		#	frame_list.append(full_frame_list[idx])
		frame_vecs = []
		for frame_id in frame_list:
			frame_vecs.append(self.visual_feat.read_one(frame_id))

		#frame_vecs = cv2.resize(np.array(frame_vecs),(2048,64))
		frames_tensor = torch.Tensor(frame_vecs)

		# text
		caption = self.captions[cap_id]
		if self.bow2vec is not None:
			cap_bow = self.bow2vec.mapping(caption)
			if cap_bow is None:
				cap_bow = torch.zeros(self.bow2vec.ndims)
			else:
				cap_bow = torch.Tensor(cap_bow)
		else:
			cap_bow = None

		if self.vocab is not None:
			tokens = clean_str(caption)
			caption = []
			caption.append(self.vocab('<start>'))
			caption.extend([self.vocab(token) for token in tokens])
			caption.append(self.vocab('<end>'))
			cap_tensor = torch.Tensor(caption)
		else:
			cap_tensor = None

		return frames_tensor, cap_tensor, cap_bow, index, cap_id, video_id, seg_id, timestamp, duration, torch.Tensor(gt_attention_score)

	def __len__(self):
		return self.length


class My_dataloader_get_input():
	"""
	Load captions and video frame features by pre-trained CNN model.
	"""

	def __init__(self, cap_file, visual_feat, bow2vec, vocab, n_caption=None, video2frames=None):
		# Captions
		self.captions = {}
		self.cap_ids = []
		self.seg_ids = []
		self.timestamps = []
		self.durations = []
		self.video_ids = set()
		self.video2frames = video2frames
		with open(cap_file, 'r') as cap_reader:
			print('cap file:', cap_file)
			for line in cap_reader.readlines():
				# try:
				cap_id, seg_id, start_time, end_time, duration, caption = line.strip().split(' ', 5)
				video_id = getVideoId(cap_id)
				video_id = video_id.replace('\xef\xbb\xbf','')
				self.captions[cap_id] = caption
				self.cap_ids.append(cap_id)
				self.seg_ids.append(int(seg_id))
				self.video_ids.add(video_id)
				self.timestamps.append((start_time,end_time))
				self.durations.append(duration)

		self.visual_feat = visual_feat
		self.bow2vec = bow2vec
		self.vocab = vocab
		self.length = len(self.cap_ids)

	def __call__(self, index):
		cap_id = self.cap_ids[index]
		seg_id = self.seg_ids[index]
		timestamp = self.timestamps[index]
		duration = self.durations[index]

		start_idx = int(1.0 * float(timestamp[0]) / float(duration) * 128)
		end_idx = int(1.0 * float(timestamp[1]) / float(duration) * 128)
		gt_attention_score = np.zeros(128)
		gt_attention_score[start_idx:end_idx] = 1

		#print('cap_id  ',cap_id)
		video_id = getVideoId(cap_id)

		# video
		#print(video_id)
		video_id = video_id.replace('\xef\xbb\xbf','').strip()
		full_frame_list = self.video2frames[video_id]
		frame_list = full_frame_list
		#idxs = random.sample(range(len(full_frame_list)),8)
		#idxs = sorted(idxs)

		#for idx in idxs:
		#	frame_list.append(full_frame_list[idx])
		frame_vecs = []
		for frame_id in frame_list:
			frame_vecs.append(self.visual_feat.read_one(frame_id))

		#frame_vecs = cv2.resize(np.array(frame_vecs),(2048,64))
		frames_tensor = torch.Tensor(frame_vecs)

		# text
		caption = self.captions[cap_id]
		if self.bow2vec is not None:
			cap_bow = self.bow2vec.mapping(caption)
			if cap_bow is None:
				cap_bow = torch.zeros(self.bow2vec.ndims)
			else:
				cap_bow = torch.Tensor(cap_bow)
		else:
			cap_bow = None

		if self.vocab is not None:
			tokens = clean_str(caption)
			caption = []
			caption.append(self.vocab('<start>'))
			caption.extend([self.vocab(token) for token in tokens])
			caption.append(self.vocab('<end>'))
			cap_tensor = torch.Tensor(caption)
		else:
			cap_tensor = None

		#return frames_tensor, cap_tensor, cap_bow, index, cap_id, video_id, seg_id, timestamp, duration, torch.Tensor(gt_attention_score)
		#return frames_tensor.unsqueeze(0), cap_tensor.unsqueeze(0), cap_bow, index, cap_id, video_id, seg_id, timestamp, duration
		videos, captions, cap_bows, idxs, cap_ids, video_ids, seg_ids, timestamps, durations = frames_tensor.unsqueeze(0), cap_tensor.unsqueeze(0), [cap_bow], index, cap_id, video_id, seg_id, timestamp, duration

		video_lengths = [min(VIDEO_MAX_LEN, len(frame)) for frame in videos]
		frame_vec_len = len(videos[0][0])
		vidoes = torch.zeros(len(videos), max(video_lengths), frame_vec_len)
		videos_origin = torch.zeros(len(videos), frame_vec_len)
		vidoes_mask = torch.zeros(len(videos), max(video_lengths))
		for i, frames in enumerate(videos):
			end = video_lengths[i]
			vidoes[i, :end, :] = frames[:end, :]
			videos_origin[i, :] = torch.mean(frames, 0)
			vidoes_mask[i, :end] = 1.0

		if captions[0] is not None:
			# Merge captions (convert tuple of 1D tensor to 2D tensor)
			lengths = [len(cap) for cap in captions]
			target = torch.zeros(len(captions), max(lengths)).long()
			words_mask = torch.zeros(len(captions), max(lengths))
			for i, cap in enumerate(captions):
				end = lengths[i]
				target[i, :end] = cap[:end]
				words_mask[i, :end] = 1.0
		else:
			target = None
			lengths = None
			words_mask = None


		cap_bows = torch.stack(cap_bows, 0) if cap_bows[0] is not None else None

		video_data = (vidoes, videos_origin, video_lengths, vidoes_mask)
		text_data = (target, cap_bows, lengths, words_mask)

		return video_data, text_data, idxs, cap_ids, video_ids, seg_ids, timestamps, durations

def get_my_data_loaders(cap_files, visual_feats, vocab, bow2vec, batch_size=100, num_workers=2, n_caption=2,
                     video2frames=None):
	return My_dataloader_get_input(cap_files['val'], visual_feats['val'], bow2vec, vocab, n_caption,
	                                    video2frames=video2frames['val'])



def get_data_loaders(cap_files, visual_feats, vocab, bow2vec, batch_size=100, num_workers=2, n_caption=2,
                     video2frames=None):
	"""
	Returns torch.utils.data.DataLoader for train and validation datasets
	Args:
		cap_files: caption files (dict) keys: [train, val]
		visual_feats: image feats (dict) keys: [train, val]
	"""
	dset = {'train': Dataset4DualEncoding(cap_files['train'], visual_feats['train'], bow2vec, vocab,
	                                      video2frames=video2frames['train']),
	        'val': Dataset4DualEncoding(cap_files['val'], visual_feats['val'], bow2vec, vocab, n_caption,
	                                    video2frames=video2frames['val'])}

	data_loaders = {x: torch.utils.data.DataLoader(dataset=dset[x],
	                                               batch_size=batch_size,
	                                               shuffle=(x == 'train'),
	                                               pin_memory=True,
	                                               num_workers=num_workers,
	                                               collate_fn=collate_frame_gru_fn)
	                for x in cap_files}
	return data_loaders


def get_test_data_loaders(cap_files, visual_feats, vocab, bow2vec, batch_size=100, num_workers=2, n_caption=2,
                          video2frames=None):
	"""
	Returns torch.utils.data.DataLoader for test dataset
	Args:
		cap_files: caption files (dict) keys: [test]
		visual_feats: image feats (dict) keys: [test]
	"""
	dset = {'test': Dataset4DualEncoding(cap_files['test'], visual_feats['test'], bow2vec, vocab, n_caption,
	                                     video2frames=video2frames['test'])}

	data_loaders = {x: torch.utils.data.DataLoader(dataset=dset[x],
	                                               batch_size=batch_size,
	                                               shuffle=False,
	                                               pin_memory=True,
	                                               num_workers=num_workers,
	                                               collate_fn=collate_frame_gru_fn)
	                for x in cap_files}
	return data_loaders


if __name__ == '__main__':
	pass