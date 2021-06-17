"""Utilities script"""

from __future__ import print_function, division
import json
import torch, os, sys, gc, shutil, time
from datetime import datetime
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import models
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from constants import COCO_SUPER_CATEGORIES, val_path, train_path,  model_path, sub_sets, coco_path

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CustomData(Dataset):
	"""Custom class inheriting dataset."""
	def __init__(self, data, transforms, phase, category):
		self.data = data
		self.transforms = transforms
		self.phase = phase
		self.category = COCO_SUPER_CATEGORIES.index(category)
		if phase == 'train' or phase == 'test':
			self.img_path = train_path
		else:
			self.img_path = val_path

	def __getitem__(self, index):
		path = self.img_path+self.data[index][0]
		target = self.transforms[self.phase](Image.open(path).convert('RGB'))
		u_label = self.data[index][1]

		# translation to 1 or 0 depending on which category_net is being trained
		if u_label == self.category:
			bin_label = 1
		else:
			bin_label = 0

		return target, bin_label

	def __len__(self):
		return len(self.data)


def divide_catcher(dividend, divisor):
	"""Just wa generic function alloing division and catching divisions by 0 so the program doesn't immediately crash."""
	try:
		res = dividend / divisor
	except Exception as e:
		res = None
	return res


def get_feature_net(model_ft, net, category="", dt_string="", prints=True):
	"""Splits up the loaded model into the feature layers and final layers for use with WFF."""
	if len(category) != 0:
		# if wanted we load in our trained fine-tuned model
		loader(model_ft, net, category, dt_string=dt_string, prints=prints)

	final_classifier = model_ft.classifier[4:] # label prediction
	model_ft.classifier = model_ft.classifier[:4] # features

	return model_ft, final_classifier


def get_net(net, pretrained=True):
	"""Get the appropriate network to load"""
	if net == 'vgg':
		model = models.vgg16(pretrained=pretrained)

		for param in model.parameters():
			param.requires_grad = False

		num_ftrs = model.classifier[6].in_features
		# set output layer to binary, 1 or 0
		model.classifier[6] = nn.Linear(num_ftrs, 2)
	else:
		print("Wrong model name")
		sys.exit()
	return model


def load_dicts(name, path):
	"""Loads json dictionary."""
	with open(path + name + '.json', 'r') as file:
		return json.load(file)


def loader(model, net, category, dt_string="", path=model_path, prints=True):
	"""Loads a model."""
	name = net + '_' + category + dt_string
	if prints:
		print("Loading model:", name)
	model.load_state_dict(torch.load(path+name+'.pth'))


def save_model(state_dict, name="coco_user_classifier", path=model_path, dt=False):
	"""Saves the trained model."""
	dt_string = ""
	if dt:
		dt_string = datetime.now().strftime("_%d-%m-%Y_%H-%M-%S")
	if not os.path.isdir(path[:-1]):
		os.mkdir(path[:-1])
	try:
		torch.save(state_dict, path + name + dt_string + '.pth')
	except:
		print("Couldn't save model at:", path+name+dt_string+'.pth, trying to save in this directory.')
		os.mkdir('./tmp_model')
		torch.save(state_dict, './tmp_model/'+name+dt_string+'.pth')


def get_transforms():
	"""Gets data transforms."""
	data_transforms = {
		'train': transforms.Compose([
			transforms.Resize(256),
			transforms.RandomResizedCrop(224),
			# transforms.CenterCrop(224),
			# transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		]),
		'val': transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		]),
		'test': transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		]),
	}
	return data_transforms


def single_user_loader(paths, category, data_transforms, batch=20, val=False):
	"""Creates a data loader specifically when working with the recipe users data for only user at a time. Mainly
	used in framework_tester.py"""
	img_data = []
	user_label = COCO_SUPER_CATEGORIES.index(category)
	for p in paths:
		img_data += [[p, user_label]]

	if val:
		phase = 'val'
	else:
		phase = 'test'

	data = CustomData(data=img_data, transforms=data_transforms, phase=phase, category=category)
	size = len(data)
	data_loader = DataLoader(data, batch_size=batch, shuffle=False,
								  num_workers=0)
	return data_loader, size


def recipe_loader(paths, truth, data_transforms, batch=20, category=""):
	"""Method that puts data into a data loader object specifically when using the data recipe format. Mainly used
	in recipe_results.py"""
	img_data = []
	for p, t in zip(paths, truth):
		img_data += [[p, t]]

	# it doesn't matter which category we pass because on looping through the dataloader we won't use the labels
	data = CustomData(data=img_data, transforms=data_transforms, phase='test', category=category)
	size = len(data)
	data_loader = DataLoader(data, batch_size=batch, shuffle=False, num_workers=0)

	return data_loader, size


def get_pos_neg_data(path, img_dir=None, headless_mode=True):
	"""Gets seperate data structures listing positive and negative data. Used in WFF."""
	users_labels = os.listdir(path)
	img_data_pos, img_data_neg = [], []
	if headless_mode or img_dir is None:
		for user_json in tqdm(users_labels):
			with open(path+user_json) as file:
				labels_user_dict = json.load(file)
			user_label = COCO_SUPER_CATEGORIES.index(labels_user_dict['user_label'][0])
			paths = labels_user_dict['paths']
			for i, p in enumerate(paths):
				if user_label == 1:
					img_data_pos += [[p, user_label]]
				else:
					img_data_neg += [[p, user_label]]

	return img_data_pos, img_data_neg


def get_list_user_data(path, img_dir=None, headless_mode=True):
	"""Creates a data structure of our data, intended to be passed to a Dataset class.
	Structure: [[path, user_label, img_id], [...], ...]

	:param path: Directory path where we can find the labels, this one is always needed. Make sure it
			finishes with a '/'.
	:param img_dir: (Optional) Img directory for local reference of images.
	:param headless_mode: Indicates whether we have an actual data folder to reference or just json files.
			On True it will only use json files to find the pictures we want in the original coco train data.
			It is recommended to use headless mode so no additional folders are needed.
	"""
	# with keys: paths, labels, ids, user_label (path being the original data train path)
	users_labels = os.listdir(path)
	img_data = []
	if headless_mode or img_dir is None:
		for user_json in tqdm(users_labels, desc="Loading data structure"):
			with open(path + user_json) as file:
				labels_user_dict = json.load(file)
			user_label = COCO_SUPER_CATEGORIES.index(labels_user_dict['user_label'][0])
			paths = labels_user_dict['paths']
			for i, p in enumerate(paths):
				img_data += [[p, user_label]]
	else:
		img_users = os.listdir(img_dir)
		for i, user in tqdm(enumerate(img_users), desc="Loading local data structure"):
			with open(path + users_labels[i]) as file:
				labels_user_dict = json.load(file)
			all_img_user = os.listdir(img_dir + user)
			user_label = COCO_SUPER_CATEGORIES.index(labels_user_dict['user_label'])
			for j, img in enumerate(all_img_user):
				path = img_dir + user + '/' + img
				img_data += [[path, user_label]]
	return img_data


def get_data_loaders(category, data_transforms, batch_size=32, num_workers=0, test=False):
	"""Gets data loaders"""
	# load data
	train_labels_dir = sub_sets+'split_train_test_val_data/' + category + '_train_data/'
	val_labels_dir = sub_sets+'split_train_test_val_data/' + category + '_val_data/'
	test_labels_dir = sub_sets+'split_train_test_val_data/' + category + '_test_data/'

	test_data = get_list_user_data(path=test_labels_dir, headless_mode=True)
	dataset_test = CustomData(data=test_data, transforms=data_transforms, phase='test', category=category)
	data_loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False,
								  num_workers=num_workers)

	if not test:
		train_data = get_list_user_data(path=train_labels_dir, headless_mode=True)
		val_data = get_list_user_data(path=val_labels_dir, headless_mode=True)

		dataset_train = CustomData(data=train_data, transforms=data_transforms, phase='train', category=category)
		dataset_val = CustomData(data=val_data, transforms=data_transforms, phase='val', category=category)

		data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,
									   num_workers=num_workers)
		data_loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False,
									 num_workers=num_workers)

		data_loaders = {'train': data_loader_train, 'val': data_loader_val, 'test': data_loader_test}
		dataset_sizes = {'train': len(dataset_train), 'val': len(dataset_val), 'test': len(dataset_test)}
	else:
		data_loaders = {'train': None, 'val': None, 'test': data_loader_test}
		dataset_sizes = {'train': 0, 'val': 0, 'test': len(dataset_test)}

	return data_loaders, dataset_sizes


def predict_one_img(file, device, model, truth, phase):
	"""Sample function for testing purposes, to predict on one individual image."""
	data_transforms = get_transforms()
	model.eval()
	model.to(device)
	if phase == 'train' or phase == 'test':
		img_path = train_path
	else:
		img_path = val_path
	path = img_path + file
	target = data_transforms[phase](Image.open(path).convert('RGB'))

	outputs = model(target.unsqueeze(0).to(device))
	_, preds = torch.max(outputs, 1)

	print('Pred:', *preds.cpu().detach().numpy().tolist())
	print('Truth', truth)


def sigmoid(z):
	"""Applies sigmoid to z."""
	return 1/(1 + torch.exp(-z))


def compute_discriminability(pred, category, net='vgg'):
	"""Computes the discriminability for weighted feature fusion.

	:param pred: The prediction of our feature model.
	:param category: The target category.
	:param net: The type of net we would like to use.
	"""
	# load calculated centroids, see compute_centroids.py
	centroid = np.load(coco_path + 'supercats_centroids/' + net + '_' + category + '.npy', allow_pickle=True)

	c_pos = torch.tensor(centroid.item()['cen_pos']).to(device)  # positive center
	c_neg = torch.tensor(centroid.item()['cen_neg']).to(device)  # negative center
	d_mean = torch.tensor(centroid.item()['mean']).to(device) # mean
	d_std = torch.tensor(centroid.item()['std']).to(device) # standard deviation

	cos_torch = nn.CosineSimilarity(dim=1, eps=1e-6) # define similarity measure
	measures = torch.cat((cos_torch(pred, c_neg).reshape(-1, 1), cos_torch(pred, c_pos).reshape(-1, 1)),
								axis=1)
	d_max = torch.max(measures, axis=1)
	d_min = torch.min(measures, axis=1)
	D_i = d_max.values / d_min.values
	D_i = sigmoid((D_i - d_mean) / d_std) # discriminability value between 0 and 1

	return D_i


# some stats functions although not used in the program currently
def calculate_single_precision(preds, truths):
	"""Computes precision, out of all predictions that our model says are positive,
	how many are actually positive.

	:param preds: list of all predictions.
	:param truths: list of all truths.
	"""
	# TP / (TP + FP)

	np_preds = np.array(preds)
	np_truths = np.array(truths)

	idx = (np_preds == 1)

	pos_pred = np_preds[idx]  # all true predictions
	corresponding_truths = np_truths[idx]  # actual corresponding label

	tp = 0
	for i in range(len(pos_pred)):
		if pos_pred[i] == corresponding_truths[i]:
			# something that was predicted positive, is actually positive, TP
			tp += 1

	prec = tp / len(pos_pred)

	return prec


def calculate_single_recall(preds, truths):
	"""Computes recall, how many actual positive cases does the model identify.
	Also known as sensitivity.

	:param preds: list of all predictions.
	:param truths: list of all truths.
	"""

	# TP / TP + FN = TP / P
	np_preds = np.array(preds)
	np_truths = np.array(truths)

	idx = (np_truths == 1)

	pos_truth = np_truths[idx]  # actual positive cases
	corresponding_preds = np_preds[idx]  # corresponding prediction

	tp = 0
	for i in range(len(pos_truth)):
		if pos_truth[i] == corresponding_preds[i]:
			# something that was predicted positive, is actually positive, TP
			tp += 1

	rec = tp / len(pos_truth)
	return rec


def calculate_fpr(preds, truths):
	"""Calculates the false positive rate.

	:param preds: list of all predictions.
	:param truths: list of all truths.
	"""
	np_preds = np.array(preds)
	np_truths = np.array(truths)

	idx = (np_truths == 0)

	neg_truth = np_truths[idx]  # all negatives
	corresponding_preds = np_preds[idx]

	fp = 0
	for p in corresponding_preds:
		if p == 1:
			fp += 1

	fpr = fp / len(neg_truth)
	return fpr


def merge_dict(dict1, dict2):
	"""Merges two dictionaries."""
	dict_merged = {}
	for (key1, value1), (key2, value2) in zip(dict1.items(), dict2.items()):
		if type(value1) is list:
			dict_merged[key1] = value1 + value2
		else:
			if key1 == 'user_label' and value1 == value2:
				dict_merged[key1] = value1
			else:
				dict_merged[key1] = [value1, value2]

	return dict_merged


def merge_folders(path, category, percentages, repeat):
	"""Merges folders to not have a very confusing file structure."""
	folder_naming = 'user_'+category+'_p'

	for r in range(repeat):
		for p in percentages:
			folder_name = folder_naming + str(p) + '_r' + str(r) + '_'
			target = folder_name + 't'
			filler = folder_name + 'f'
			# merge target and filler folders
			new_folder = folder_name[:-1]
			new_path = path + new_folder
			old_path_t = path + target
			old_path_f = path + filler
			# create new folder for merging
			if os.path.isdir(new_path):
				shutil.rmtree(new_path)
				time.sleep(1)
			os.mkdir(new_path)
			# copy target and filler to new folder
			time.sleep(1)
			shutil.copytree(old_path_t, new_path + '/target_imgs/')
			shutil.copytree(old_path_f, new_path + '/filler_imgs/')
			# remove old folders
			if os.path.isdir(old_path_t):
				shutil.rmtree(old_path_t)
			if os.path.isdir(old_path_f):
				shutil.rmtree(old_path_f)


def dump_json(dictionary, name, path):
	"""Dumps json file content. """
	last = len(path.split('/')[-2])+1
	if not os.path.isdir(path[:-last]):  # check whether the folder one higher up exists
		os.mkdir(path[:-last])
	if not os.path.isdir(path):
		os.mkdir(path)
	with open(path + name + '.json', 'w+') as file:
		json.dump(dictionary, file)


# MV
def majority_vote(profile_loader, net='vgg', prints=False, include_negative=False, threshold=None):
	"""Checks the majority vote results for all supercategories.

	:param profile_loader: DataLoader for one profile
	:param str net: Network we would like to use
	:param bool prints: Whether to show informative print statements.
	:param bool include_negative: To include negative predictions as well when returning. Otherwise it only shows positive
		predictions. We make the implicit assumption when getting results that if it doesn't show up the prediction was
		negative.
	:param float threshold: Floating point for picking the threshold of when we should predict true. If None then we
		decide based on majority.
	"""
	predictions_per_cat = {}
	for sc in COCO_SUPER_CATEGORIES:
		# get pretrained model and load dict state
		model = get_net(net)
		loader(model, net, sc, dt_string='_ep24', prints=prints)
		# set to eval and to device
		model.eval()
		model.to(device)

		all_preds = []
		for inputs, _ in profile_loader:
			gc.collect()
			torch.cuda.empty_cache()
			inputs = inputs.to(device)

			# get outputs from model
			outputs = model(inputs)
			# convert to them to binary, 1 and 0
			_, preds = torch.max(outputs, 1)
			all_preds += preds.cpu().detach().numpy().tolist()

		dct = {1:0, 0:0}
		for pred in all_preds:
			# count the amount a certain prediction was made, e.g. 1: 4 times, 0: 2 times
			dct[pred] = all_preds.count(pred)

		if threshold:
			amount = int(threshold * len(all_preds)) # amount of pictures to be target
			if dct[1] >= amount:
				confidence = dct[1] / len(all_preds)
				prediction = True
				predictions_per_cat[sc] = (prediction, confidence)
			else:
				prediction = False
				if include_negative:
					predictions_per_cat[sc] = (prediction, confidence)
		else:
			# pick the most occurring prediction
			user_label, occurrence = max(dct.items(), key=lambda x: x[1])
			# calculate confidence value
			confidence = occurrence / len(all_preds)

			# convert 1 and 0 to true and false labels for less ambiguity, add to dictionary + confidence
			if user_label == 1:
				prediction = True
				predictions_per_cat[sc] = (prediction, confidence)
			else:
				prediction = False
				if include_negative:
					predictions_per_cat[sc] = (prediction, confidence)

		if prints:
			print("Prediction for", sc, ":", prediction, "with confidence:", confidence)

	return predictions_per_cat


# WFF
def wff(profile_loader, net='vgg', include_negative=False, threshold=0):
	"""Applies the Weighted Feature Fusion classifier to one profile.

	:param profile_loader: DataLoader of one singular profile.
	:param str net: Net we would like to use.
	:param bool include_negative: To include negative predictions as well when returning. Otherwise it only shows positive
		predictions. We make the implicit assumption when getting results that if it doesn't show up the prediction was
		negative.
	"""

	predictions_per_cat = {}

	for sc in COCO_SUPER_CATEGORIES:
		# load model and split up the loaded dict state
		model = get_net(net)
		model_ft, final_classifier = get_feature_net(model, net, sc, dt_string='_ep24', prints=False)

		# set to eval and device
		model_ft.eval()
		model_ft.to(device)
		final_classifier.to(device)

		data = []
		labels_all = []
		for inputs, labels in profile_loader:
			# collect all inputs and labels in one tensor, pass that tensor to compute
			# inputs of size (batch_size, 3, 224, 224) append to tensor of (N, 3, 224, 244), with N amount of images
			labels_all += labels
			data += inputs
		data = torch.stack(data)
		data.reshape((len(data), 3, 224, 224))  # reshape into proper size

		# get the features from the feature layers of the model
		pred = model_ft(data.to(device))
		D_i = compute_discriminability(pred, sc, net)  # compute D_i
		group_feature = pred.cpu().detach()

		# calculate K and R, see <https://arxiv.org/abs/2008.10850>
		K = (1 / (D_i.max() - D_i.min()))
		R = K * D_i + (1 - K * torch.max(D_i))

		# logical index to pick R values above threshold
		idx = (R > threshold)
		group_rep = torch.mean((R[idx].reshape((-1, 1)).to(device) * group_feature[torch.where(idx == True)[0]]
								.to(device)), axis=0)

		# get prediction by passing weighted features through the final classifier
		label = (final_classifier(group_rep.to(device).float()).argmax(0)).cpu().float().numpy()
		# R = R.cpu().detach().numpy().tolist()
		if int(label.tolist()) == 1:
			# dct[sc] = (True, R)
			predictions_per_cat[sc] = (True, label.tolist())
		else:
			# dct[sc] = (False, R)
			if include_negative:
				predictions_per_cat[sc] = (False, label.tolist())

	return predictions_per_cat
