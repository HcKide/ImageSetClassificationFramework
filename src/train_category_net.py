"""Trains a network per supercategory."""

from __future__ import print_function, division
import torch
import time
import gc
import copy
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils import save_model, loader, get_net, get_transforms, get_data_loaders, \
	calculate_single_recall, calculate_single_precision, calculate_fpr
from constants import COCO_SUPER_CATEGORIES


phases = ['train', 'val', 'test']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(model, data_loaders, data_sizes, category, net, lr=0.0001, num_epochs=12):
	"""Trains the model."""

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.classifier.parameters(), lr=lr, momentum=0.9)
	scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

	since = time.time()
	best_acc = 0.0

	print("Training model for category:", category)
	print()

	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch+1, num_epochs))
		print('-' * 20)
		for phase in phases[:-1]:
			if phase == 'train':
				model.train()
			else:
				model.eval()

			running_loss = 0.0
			running_corrects = 0

			# iterate
			for inputs, labels in data_loaders[phase]:
				gc.collect()
				torch.cuda.empty_cache()
				inputs = inputs.to(device)
				labels = labels.to(device)

				optimizer.zero_grad()

				with torch.set_grad_enabled(phase == 'train'):
					outputs = model(inputs)
					_, preds = torch.max(outputs, 1)
					loss = criterion(outputs, labels)

					if phase == 'train':
						loss.backward()
						optimizer.step()

				# statistics
				running_loss += loss.item() * inputs.size(0)
				running_corrects += torch.sum(preds == labels.data)

			if phase == 'train':
				scheduler.step()

			epoch_loss = running_loss / data_sizes[phase]
			if phase == 'val':
				epoch_acc = running_corrects.double() / data_sizes[phase]
				print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

			if phase == 'val' and epoch_acc > best_acc:
				best_acc = epoch_acc
				best_model_wts = copy.deepcopy(model.state_dict())  # saves model with best val accuracy

	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(
		time_elapsed // 60, time_elapsed % 60))
	print('Best val Acc: {:4f}'.format(best_acc))

	# saves model and writes to disk
	save_model(best_model_wts, net + '_' + category + '_ep' + str(num_epochs))


def data_tester(model, data_loaders, data_sizes, extra_stats=True):
	"""Tests the data."""
	corrects = 0
	all_labels = []
	all_preds = []
	for inputs, labels in tqdm(data_loaders['test'], desc="Testing"):
		gc.collect()
		torch.cuda.empty_cache()
		inputs = inputs.to(device)
		labels = labels.to(device)
		all_labels += labels.cpu().detach().numpy().tolist()

		outputs = model(inputs)
		_, preds = torch.max(outputs, 1)
		all_preds += preds.cpu().detach().numpy().tolist()

		corrects += torch.sum(preds == labels.data)

	epoch_acc = corrects.double() / data_sizes['test']
	print('Test Acc: {:4f}%'.format(epoch_acc*100))

	if extra_stats:

		recall = calculate_single_recall(all_preds, all_labels)
		precision = calculate_single_precision(all_preds, all_labels)
		fpr = calculate_fpr(all_preds, all_labels)
		print("Recall: {:4f}% Precision: {:4f}% and FPR: {:4f}".format(recall*100, precision*100, fpr*100))

	return epoch_acc


def main(category, net='vgg', test=False):
	"""Main entry function.

	:param str category: Category to train/test for.
	:param str net: Network to use. Default is 'vgg'.
	:param bool test: Toggle for testing. When off we train.
	"""
	# data transforms
	data_transforms = get_transforms()

	model = get_net(net)
	model.to(device)

	data_loaders, dataset_sizes = get_data_loaders(category, data_transforms, batch_size=16, test=test)

	if test:
		loader(model, net, category, dt_string='_ep24')
		# print("Test size", dataset_sizes['test'])
		acc = data_tester(model, data_loaders, dataset_sizes)
	else:
		train_model(model, data_loaders, dataset_sizes, category=category, net=net)


if __name__ == '__main__':

	# for sc in COCO_SUPER_CATEGORIES:  # train for all categories one by one
	# 	main(sc, test=True)

	main('accessory', test=True)  # train for one category
