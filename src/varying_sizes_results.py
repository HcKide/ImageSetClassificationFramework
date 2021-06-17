"""Script for testing users of varying sizes. Not used in research paper so functions might be old and need to be
adjusted, included for completeness. """

from constants import COCO_SUPER_CATEGORIES, sub_sets
from tqdm import tqdm
from utils import load_dicts, majority_vote, wff, get_transforms, recipe_loader
import time, json, os


def varying_size_mv():
	"""Tests various sizes user profiles, for Majority Voting. Not used in research paper."""
	since = time.time()

	sizes = ['10', '20', '30', '40', '50']
	scs = COCO_SUPER_CATEGORIES

	path = sub_sets + '/artificial_users/varying_size_users/'
	res_path = path + 'results_mv/'

	if type(scs) is not list:
		scs = [scs]

	net = 'vgg'
	data_transforms = get_transforms()

	for sc in scs:

		recipe_dict = load_dicts(name='users_' + sc, path=path)
		users_per_cat = {}
		for s in tqdm(sizes, desc="Using supercategory " + sc):
			user = recipe_dict[s]
			truth = user['labels']
			paths = user['paths']

			# get data_loader object for one user at a time
			data_loader, size = recipe_loader(paths, truth, data_transforms, category=sc)

			prediction_data = majority_vote(data_loader, net)
			users_per_cat[s] = prediction_data

		if not os.path.isdir(res_path):
			os.mkdir(res_path)
		with open(res_path + 'varying_size_' + sc + '_results.json', 'w+') as file:
			json.dump(users_per_cat, file)

	time_elapsed = time.time() - since
	print('Results complete in {:.0f}m {:.0f}s'.format(
		time_elapsed // 60, time_elapsed % 60))


def varying_size_wff():
	"""Tests various sizes user profiles, for Weighted Feature Fusion. Not used in research paper."""
	since = time.time()

	sizes = ['10', '20', '30', '40', '50']
	scs = COCO_SUPER_CATEGORIES

	path = sub_sets + '/artificial_users/varying_size_users/'
	res_path = path + 'results_wff/'

	if type(scs) is not list:
		scs = [scs]

	net = 'vgg'
	data_transforms = get_transforms()

	for sc in scs:
		thresholds_dct = load_dicts('wff_thresholds', sub_sets + 'thresholds/')
		threshold = thresholds_dct[sc]

		recipe_dict = load_dicts(name='users_' + sc, path=path)
		users_per_cat = {}
		for s in tqdm(sizes, desc="Using supercategory " + sc):
			user = recipe_dict[s]
			truth = user['labels']
			paths = user['paths']

			# get data_loader object for one user at a time
			data_loader, size = recipe_loader(paths, truth, data_transforms, category=sc)

			prediction_data = wff(data_loader, net, threshold=threshold)
			users_per_cat[s] = prediction_data

		if not os.path.isdir(res_path):
			os.mkdir(res_path)
		with open(res_path + 'varying_size_' + sc + '_results.json', 'w+') as file:
			json.dump(users_per_cat, file)

	time_elapsed = time.time() - since
	print('Results complete in {:.0f}m {:.0f}s'.format(
		time_elapsed // 60, time_elapsed % 60))


if __name__ == '__main__':
	# varying_size_wff()
	varying_size_mv()
