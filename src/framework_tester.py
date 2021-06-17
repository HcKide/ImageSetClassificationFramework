"""Script for testing the entire framework"""

from utils import wff, majority_vote, load_dicts, get_transforms, single_user_loader, divide_catcher
from constants import COCO_SUPER_CATEGORIES, sub_sets
import torch, os, json, time
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def data_tester_p_users(category, method, percentage='10', path=sub_sets + 'threshold_users/10p/', net='vgg',
                        threshold_dict=False, dump=True, val=False, threshold=None, show_stats=False, disable=False,
                        all_files=False, single_object=None):
	"""Method that tests final classifiers MV and WFF but for data that's in the format of a data recipe.
	This way we can test for sets of data where a user profile is considered True, at a certain threshold of content,
	rather than the 'pure' form.

	:param str category: The target category, it will be treated as the truth label for all users we're using. This way we
		can calculate the accuracy per category for all models.
	:param function method: The method we want to use, it support either wff() or majority_vote().
	:param str percentage: Percentage to load. Has effect on the pictures loaded and threshold loaded.
	:param str path: Path to use.
	:param str net: Standard set as 'vgg', it's the type of network we would like to use.
	:param bool threshold_dict: Boolean indicating whether the threshold dictionary should be used.
	:param bool dump: Whether to dump the accuracies json dictionary or not.
	:param bool val: Whether to use validation data or not.
	:param float threshold: Threshold value only passed if not None. Only works with the method wff(). Between 0-1.
	:param bool show_stats: Boolean flag indicating whether to include extra stats in the returned dictionary.
	:param bool disable: Disable the first tqdm bar, especially useful when only focusing on one category at a time.
	:param bool all_files: Use all json files in the path. Default is False. It then only picks the files that are of
		the category we'd like to investigate.
	:param str single_object: Parameter indicating whether a single object is used. Since it uses different
		thresholds it needs to be specified when loading them.
	"""
	since = time.time()
	data_transforms = get_transforms()

	# define path where the test partitions are located
	if val:
		test_path = path + 'val/'
	else:
		test_path = path + 'test/'

	method_str = ""
	if method.__name__ == 'majority_vote':
		method_str = "mv"
	elif method.__name__ == 'wff':
		method_str = "wff"

	if all_files:
		# get list of all files in that test path
		all_dcts = os.listdir(test_path)
	else:
		all_dcts = []
		files = os.listdir(test_path)
		for f in files:
			cat = f.split('_')[1].split('.')[0]
			if category == cat:
				all_dcts.append(f)
			elif single_object:
				if single_object == cat:
					all_dcts.append(f)

	t_str = ""
	if threshold_dict:
		if single_object:
			method_str += '_' + single_object
		thresholds_dct = load_dicts(method_str+'_thresholds_'+percentage+'p', sub_sets + 'thresholds/')
		t_nr = thresholds_dct[category]
		t_str = " with threshold " + str(t_nr)

	if threshold is not None:
		t_nr = threshold
		t_str = " with threshold " + str(t_nr)

	accs_p_cat = {}
	for super_cat_dct in tqdm(all_dcts, desc=method_str + ": " + percentage + "% " + category + " users" + t_str,
	                          disable=disable):
		# load the user into a dictionary
		user_dict = load_dicts(name=super_cat_dct.split('.')[0], path=test_path)
		all_users = len(user_dict[percentage])

		# pass them and the target category to a recipe loader function which returns a data_loader for that user
		tp, fp, fn, tn = 0, 0, 0, 0
		if disable:
			desc = method_str + ": " + percentage + "% " + category + " users" + t_str
		else:
			desc = ""
		for i in tqdm(range(all_users), desc=desc):
			paths = user_dict[percentage][i]['paths']
			u_label = user_dict[percentage][i]['user_label']

			data_loader, size = single_user_loader(paths, category, data_transforms, val=val)
			# use the passed method to predict user labels
			if threshold_dict or threshold:
				prediction_data = method(data_loader, net, threshold=t_nr)
			else:
				prediction_data = method(data_loader, net)

			keys = prediction_data.keys()
			if category in keys:
				# if the target is found in all positive predictions
				if u_label == category:
					# true prediction when true
					tp += 1
				else:
					# true prediction when false
					fp += 1
			else:
				# if the target is not found in all positive predictions
				if u_label == category:
					# false prediction when true
					fn += 1
				else:
					# false prediction when false
					tn += 1

		# calculates stats for all users per dictionary
		accuracy = divide_catcher(tp + tn, all_users)
		fpr = divide_catcher(fp, fp+tn)
		fnr = divide_catcher(fn, fn + tp)
		sensitivity = divide_catcher(tp, tp + fn)
		specificity = divide_catcher(tn, tn + fp)

		accs = {}
		accs[category] = accuracy
		accs['user_nr'] = all_users
		accs['img_nr'] = len(paths)
		accs['threshold'] = t_nr
		accs['method'] = method_str
		if show_stats:
			accs['fpr'] = fpr
			accs['fnr'] = fnr
			accs['sensitivity'] = sensitivity
			accs['specificity'] = specificity
			accs['tp'] = tp
			accs['fp'] = fp
			accs['fn'] = fn
			accs['tn'] = tn
		accs_p_cat[super_cat_dct.split('.')[0]] = accs

	if dump:
		if val:
			results_path = path + 'val_results/'
		else:
			results_path = path + 'test_results/'
		if not os.path.isdir(results_path):
			os.mkdir(results_path)
		file = open(results_path + method_str+'_accuracies_' + str(all_users) + 'users_' + category + '.json', 'w+')
		json.dump(accs_p_cat, file)
		file.close()

	time_elapsed = time.time() - since
	print('Results complete in {:.0f}m {:.0f}s'.format(
		time_elapsed // 60, time_elapsed % 60))

	return accs_p_cat


def data_tester(category, method, net='vgg', prints=False, threshold_dict=False, slice_end=None, dump=True, val=False,
                threshold=None):
	"""Tests the final classifiers MV and WFF and calculates their accuracies over a set of users. Not used for any
	results in the paper.

	:param str category: The target category, it will be treated as the truth label for all users we're using. This way we
		can calculate the accuracy per category for all models.
	:param method: The method we want to use, it support either wff() or majority_vote()

	:param str net: Standard set as 'vgg', it's the type of network we would like to use.
	:param bool prints: Indicates whether to display informative print statements.
	:param bool threshold_dict: Boolean indicating whether the threshold dictionary should be used.
	:param int slice_end: Integer indicating the end of the slice for using less users than original set.
	:param bool dump: Whether to dump the json file or not.
	:param bool val: Whether to use validation data or not.
	:param float threshold: Threshold value only passed if not None. Only works with the method wff(). Between 0-1.
	"""
	since = time.time()
	data_transforms = get_transforms()
	results_path = sub_sets + 'test_results/'

	# define path where the test partitions are located
	if val:
		test_path = sub_sets + 'special_testing_data/' + category + '_val/'
	else:
		test_path = sub_sets + 'special_testing_data/' + category + '_test/'

	# get list of all users in that test path
	all_users = os.listdir(test_path)

	if slice_end:
		all_users = all_users[:slice_end]

	# create a dictionary that counts the correct predictions for each category

	t_str = ""
	if threshold_dict:
		thresholds_dct = load_dicts('wff_thresholds', sub_sets + 'thresholds/')
		t_nr = thresholds_dct[category]
		t_str = " with threshold " + str(t_nr)

	if threshold:
		t_nr = threshold
		t_str = " with threshold " + str(t_nr)

	counter_dct = {sc: 0 for sc in COCO_SUPER_CATEGORIES}
	for user in tqdm(all_users, desc=category + " users" + t_str):
		# load the user into a dictionary
		user_dict = load_dicts(name=user.split('.')[0], path=test_path)
		paths = user_dict['paths']  # get the file names / paths

		# pass them and the target category to a recipe loader function which returns a data_loader for that user
		data_loader, size = single_user_loader(paths, category, data_transforms, batch=20, val=val)

		# used the passed method to predict user labels
		if threshold_dict or threshold:
			prediction_data = method(data_loader, net, threshold=t_nr)
		else:
			prediction_data = method(data_loader, net)

		incorrects = []
		target_met = False
		# loop through all keys found in our prediction data, i.e. everywhere it predicted True
		for k in prediction_data.keys():
			prd = prediction_data[k][0]
			if k == category:
				# if the True category is our target category, we can say it predicted correctly, True Positive
				target_met = True
				if prints:
					print("Correct prediction:", k, prd)
			else:
				# else, if the True category is not our target category, we can say it is predicted incorrectly, False Positive
				incorrects.append(k)
				if prints:
					print("Incorrect prediction:", k, prd, "when it is not.")

		for sc in COCO_SUPER_CATEGORIES:
			# add all correct predictions
			if sc == category and target_met:
				# if we see the target was encountered as a True Positive, we add one to the correctness counter
				counter_dct[sc] = counter_dct[sc] + 1
			elif sc not in incorrects:
				# if the sc is not found in incorrects it was correctly recognized as True negative
				counter_dct[sc] = counter_dct[sc] + 1
		# if neither of these conditions trigger, the prediction was wrong and we don't add to correctness count

	# calculate accuracies
	accs = {}
	for k in counter_dct.keys():
		acc = counter_dct[k] / len(all_users)
		if prints:
			print("Accuracy for", k, "net:", counter_dct[k] / len(all_users))
		accs[k] = acc
	mean_acc = sum(accs.values()) / len(accs.values())

	accs['user_nr'] = len(all_users)
	accs['mean_acc'] = mean_acc
	accs['img_nr'] = len(user_dict['paths'])

	if dump:
		if method.__name__ == 'wff':
			file = open(results_path + 'wff_accuracies_' + str(len(all_users)) + 'users_' + category + '.json', 'w+')
		elif method.__name__ == 'majority_vote':
			file = open(results_path + 'mv_accuracies_' + str(len(all_users)) + 'users_' + category + '.json', 'w+')
		json.dump(accs, file)
		file.close()

	time_elapsed = time.time() - since
	print('Results complete in {:.0f}m {:.0f}s'.format(
		time_elapsed // 60, time_elapsed % 60))

	return accs


def calculate_thresholds(method, path=sub_sets + 'threshold_users/', recipe=False, target=None, percentage=None, start=0,
                         end=10, single_category=None):
	"""Calculates thresholds on validation data.

	:param function method: Method to be used to calculate accuracies. Works with wff and majority_vote.
	:param str path: Path to load users from.
	:param bool recipe: Boolean indicating whether the recipe json format is used. It then compares the target class
		accuracies rather than the mean average.
	:param str target: Target category we would like to optimize accuracy for. Only used when recipe is True. Needs
		to be set if recipe is True.
	:param str percentage: Percentage indicator used in loading the correct path and passing to the function used when
		recipe is True. Needs to be set if recipe is True.
	:param int start: Start value between 0 and 10. Allows skipping some thresholds values for faster optimizing.
		Default value is 0 threshold of 0.0
	:param int end: End value between 0 and 10, should be higher than start value. Default is 10, i.e. threshold of 1.0
	:param str single_category: The single category needs to be specified when used. Otherwise leave default None.
		It's mainly used in naming the threshold folder correctly, as it uses a different one than standard.
	"""

	thresholds = {}

	if percentage:
		path = path+percentage+'/'

	if recipe:
		scs = [target]
	else:
		scs = COCO_SUPER_CATEGORIES
	for sc in scs:
		best_acc = 0
		best_t = 0
		for i in range(start, end):
			threshold = i / 10
			if recipe:
				acc = data_tester_p_users(sc, method, percentage[:-1], path=path, val=True, threshold=threshold, dump=False,
				                          show_stats=True, disable=True)
				print(acc)
			else:
				acc = data_tester(sc, method, val=True, threshold=threshold, dump=False)
			if recipe:
				for k in acc.keys():
					accuracy = acc[k][target]

					if accuracy >= best_acc:
						print("Old acc/t", best_acc, best_t)
						best_acc = accuracy
						best_t = threshold
						print("New acc/t:", best_acc, best_t)
			else:
				if acc['mean_acc'] >= best_acc:
					print("Old mean acc/t", best_acc, best_t)
					best_acc = acc['mean_acc']
					best_t = threshold
					print("New mean acc/t:", best_acc, best_t)
		print("Best threshold for", sc, ":", best_t, "with accuracy:", best_acc)
		thresholds[sc] = best_t

	t_path = sub_sets + 'thresholds/'
	if method.__name__ == 'wff':
		m_str = 'wff'
	elif method.__name__ == 'majority_vote':
		m_str = 'mv'
	else:
		print("Method not recognized")
		m_str = 'unknown'

	if single_category:
		m_str += '_' + single_category

	file_path = t_path + m_str + '_thresholds_'+percentage+'.json'
	if os.path.isfile(file_path):
		# if file already exists, merge the old and new
		dct = load_dicts(file_path.split('/')[-1].split('.')[0], t_path)
		new_ks = thresholds.keys()
		for k, v in dct.items():
			if k not in new_ks:
				thresholds[k] = v

	with open(file_path, 'w+') as file:
		json.dump(thresholds, file)


def main(distribution_results=False, snowboard=False, toilet=False):
	if distribution_results:
		ps = ['10', '60', '90']
		target = 'sports'
		# for p in ps:
		# 	if p == '10':
		# 		start = 0
		# 		end = 3
		# 	elif p == '60':
		# 		start = 3
		# 		end = 8
		# 	elif p == '90':
		# 		start = 4
		# 		end = 10
		# 	calculate_thresholds(majority_vote, recipe=True, target=target, percentage="90p", start=start, end=end)

		# for p in ps:
		# 	calculate_thresholds(wff, recipe=True, target=target, percentage=p + "p")

		# wff
		for p in ps:
			accs = data_tester_p_users(target, wff, path=sub_sets + 'threshold_users/' + p + 'p/', percentage=p,
			                           val=False, threshold_dict=True, dump=True, show_stats=True, disable=True)
			print(accs)

		# mv
		for p in ps:
			accs = data_tester_p_users(target, majority_vote, path=sub_sets + 'threshold_users/' + p + 'p/',
			                           percentage=p,
			                           val=False, threshold_dict=True, dump=True, show_stats=True, disable=True)
			print(accs)

	if snowboard:
		ps = ['10', '60', '90']
		target = 'sports'
		single_t = 'snowboard'
		# for p in ps:
		# 	if p == '10':
		# 		start = 0
		# 		end = 3
		# 	elif p == '60':
		# 		start = 5
		# 		end = 8
		# 	elif p == '90':
		# 		start = 4
		# 		end = 10
		# 	# mv threshold calculation
		# 	calculate_thresholds(majority_vote, recipe=True, target=target, percentage=p + "p", start=start, end=end,
		# 	                     single_category=single_t)

		# for p in ps:
		# 	# wff threshold calculation
		# 	calculate_thresholds(wff, recipe=True, target=target, percentage=p + "p", single_category=single_t)

		# wff
		for p in ps:
			accs = data_tester_p_users(target, wff, path=sub_sets + 'threshold_users/' + p + 'p_'+single_t+'/', percentage=p,
			                           val=False, threshold_dict=True, dump=True, show_stats=True, disable=True, single_object=single_t)
			print(accs)

		# mv
		for p in ps:
			accs = data_tester_p_users(target, majority_vote, path=sub_sets + 'threshold_users/' + p + 'p_'+single_t+'/',
			                           percentage=p, val=False, threshold_dict=True, dump=True, show_stats=True, disable=True,
			                           single_object=single_t)
			print(accs)

	if toilet:
		ps = ['10', '60', '90']
		target = 'furniture'
		single_t = 'toilet'
		# for p in ps:
		# 	if p == '10':
		# 		start = 0
		# 		end = 3
		# 	elif p == '60':
		# 		start = 4
		# 		end = 8
		# 	elif p == '90':
		# 		start = 6
		# 		end = 10
		# 	# mv threshold calculation
		# 	if p != '10':
		# 		calculate_thresholds(majority_vote, recipe=True, target=target, percentage=p + "p", start=start, end=end,
		# 		                     single_category=single_t)

		# for p in ps:
		# 	if p == '10':
		# 		start = 0
		# 		end = 5
		# 	elif p == '60':
		# 		start = 0
		# 		end = 6
		# 	elif p == '90':
		# 		start = 2
		# 		end = 9
		# 	# wff threshold calculation
		# 	calculate_thresholds(wff, recipe=True, target=target, percentage=p + "p", single_category=single_t,
		# 	                     start=start, end=end)

		# wff
		for p in ps:
			accs = data_tester_p_users(target, wff, path=sub_sets + 'threshold_users/' + p + 'p_'+single_t+'/', percentage=p,
			                           val=False, threshold_dict=True, dump=True, show_stats=True, disable=True, single_object=single_t)
			print(accs)

		# mv
		for p in ps:
			accs = data_tester_p_users(target, majority_vote, path=sub_sets + 'threshold_users/' + p + 'p_'+single_t+'/',
			                           percentage=p, val=False, threshold_dict=True, dump=True, show_stats=True, disable=True,
			                           single_object=single_t)
			print(accs)


if __name__ == '__main__':
	main(toilet=True, snowboard=True)
