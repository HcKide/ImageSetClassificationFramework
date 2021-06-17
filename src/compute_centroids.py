from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from utils import get_transforms, get_pos_neg_data, get_feature_net, get_net
from constants import COCO_SUPER_CATEGORIES, sub_sets, train_path, coco_path
import os, time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
plt.ion()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
	"""Computes the centroids per super-category, used in WFF classification."""
	since = time.time()

	data_transforms = get_transforms()
	nets = ['vgg']
	scs = COCO_SUPER_CATEGORIES
	if type(scs) is not list:
		scs = [scs]

	for net in nets:
		for sc in scs:
			# get feature model
			model_ft = get_net(net)
			model_ft, _ = get_feature_net(model_ft, net, sc, dt_string='_ep24')
			model_ft.eval()
			model_ft.to(device)

			train_labels_dir = sub_sets + 'split_train_test_val_data/' + sc + '_train_data/'
			img_data_pos, img_data_neg = get_pos_neg_data(train_labels_dir)

			pos_feature = np.zeros((len(img_data_pos), 4096))
			for i in range(len(img_data_pos)):
				pos_feature[i] = model_ft(
					data_transforms['val'](Image.open(train_path+img_data_pos[i][0]).convert('RGB')).unsqueeze(0).cuda()).reshape(
					(1, -1)).cpu().detach().numpy()

			neg_feature = np.zeros((len(img_data_neg), 4096))
			for i in range(len(img_data_neg)):
				neg_feature[i] = model_ft(
					data_transforms['val'](Image.open(train_path+img_data_neg[i][0]).convert('RGB')).unsqueeze(0).cuda()).reshape(
					(1, -1)).cpu().detach().numpy()

			anchor_feat = np.append(np.mean(pos_feature, axis=0).reshape(1, -1),
									np.mean(neg_feature, axis=0).reshape(1, -1), axis=0)
			anchor_feat /= np.linalg.norm(anchor_feat, axis=1).reshape(-1, 1)

			feat = np.append(pos_feature, neg_feature, axis=0)
			feat /= np.linalg.norm(feat, axis=1).reshape(-1, 1)

			discriminability = np.ones(feat.shape[0], dtype=np.float32) * -1

			sub_feats = feat[:len(img_data_pos), :]
			cos_matrix = sub_feats.dot(anchor_feat.T)
			discriminability_vector = cos_matrix[:, 0] / cos_matrix[:, 1]
			discriminability[:len(img_data_pos)] = discriminability_vector

			sub_feats = feat[len(img_data_pos):, :]
			cos_matrix = sub_feats.dot(anchor_feat.T)
			discriminability_vector = cos_matrix[:, 1] / cos_matrix[:, 0]
			discriminability[len(img_data_pos):] = discriminability_vector

			centr_dict = {}
			centr_dict['cen_pos'] = np.mean(pos_feature, axis=0).reshape(1, -1)
			centr_dict['cen_neg'] = np.mean(neg_feature, axis=0).reshape(1, -1)
			centr_dict['mean'] = discriminability.mean()
			centr_dict['std'] = discriminability.std()

			centroid_path = coco_path + 'supercats_centroids/'
			if not os.path.isdir(centroid_path):
				os.mkdir(centroid_path)
			np.save(centroid_path + net + '_' + sc + '.npy', centr_dict)

	time_elapsed = time.time() - since
	print('Results complete in {:.0f}m {:.0f}s'.format(
		time_elapsed // 60, time_elapsed % 60))


if __name__ == '__main__':
	main()
