"""Functions for creating users using the coco data. """

import random, os, shutil, time, json
from constants import sub_sets


class UserSampling:
	"""Class containing functions for creating users. """

	def __init__(self, coco, labels_path=sub_sets, imgs_path=None):
		self.coco = coco
		self.labels_path = labels_path if labels_path[-1] == '/' else labels_path + '/'
		if imgs_path:
			self.data_path = imgs_path if imgs_path[-1] == '/' else imgs_path + '/'
		else:
			self.data_path = self.labels_path + 'imgs/'

	def create_user(self, name, category: str or list, super_cat=False, size=20,
						   toggle_folder_creation=False, dump=True, suppress_print=False,
						   anti_category: str or list=None, super_cat_anti=False, custom_user_label=None, split=False,
						   split_name=""):
		"""Generic function for creating users.

		:param str name: Name of the user.
		:param str/list category: The category/-ies we want to sample from.
		:param bool super_cat: Indicating whether the category parameter is of type super-category. If yes -> indicate
			True. Cannot mix super-cats and non-supercats in one create_user call. If it is desired create two create_user
			calls and merge the dictionaries in the end.
		:param int size: Size of the image set. Default is 20.
		:param bool toggle_folder_creation: Boolean toggling whether we would like to locally copy the pictures of the
			users to img folders, that way we can easily inspect the actual images rather than their entries in a
			dictionary which is what the create_user method returns. Default is False.
		:param bool dump: Boolean flag indicating whether we want to dump the dictionary to a json file and write to the
			disk. Standard is True.
		:param bool suppress_print: Boolean flag, set to true for suppressing print statements.
		:param str/list anti_category: The category/-ies we do not want to sample from at all costs. Optional, default
			is None.
		:param bool super_cat_anti: Boolean flag indicating whether the anti-categories are super-categories.
		:param str custom_user_label: Allows giving the user a custom_user_label. Otherwise the category will be the
			chosen user_label.
		:param bool split: Indicating whether we sample from one of the split partitions, train or test split.
		:param str split_name: If split=True, indicate the split_name: 'split_test' / 'split_train'.

		:return dictionary
		"""

		if super_cat:
			if type(category) != list:
				category = [category]
			filter_int = []
			for c in category:
				filter_int += self.coco.super_cat_ids[c]
			filters = [self.coco.all_categories[c][0] for c in filter_int]
			filters = list(set(filters))
		else:
			if type(category) is list:
				filters = category
			else:
				filters = [category]

		all_labels = []
		images_labels = []
		for f in filters:
			if split:
				img_names, labels = self.coco.load_split_imgs(split_name, [f])
			else:
				img_names, labels = self.coco.load_images_with_filter([f], suppress_print=suppress_print)
			all_labels += labels
			for img, l in zip(img_names, labels):
				images_labels += [[img, l]]

		if anti_category is not None:
			if super_cat_anti:
				if type(anti_category) != list:
					anti_category = [anti_category]
				anti_filter_int = []
				for ac in anti_category:
					anti_filter_int += self.coco.super_cat_ids[ac]
				anti_filter = [self.coco.all_categories[c][0] for c in anti_filter_int]
			else:
				if type(anti_category) is list:
					anti_filter = anti_category
				else:
					anti_filter = [anti_category]

			del_indices = []
			for i, l in enumerate(all_labels):
				for af in anti_filter:
					af_cat_id = self.coco.coco.getCatIds(catNms=af)
					if af_cat_id[0] in l:
						# remove the item from images_labels
						del_indices.append(i)

			count = 0
			del_indices = sorted(list(set(del_indices)))
			if not suppress_print:
				print("Removing:", len(del_indices))
			for i in del_indices:
				del images_labels[i - count]
				count += 1

		if custom_user_label:
			user_label = custom_user_label
		else:
			user_label = category

		dict_user = self.sample_pictures(images_labels, size, toggle_folder_creation=toggle_folder_creation, name=name,
										 user_label=user_label, dump=dump)
		return dict_user

	def get_all_images_category(self, super_cat, suppress_print=False):
		"""Returns all images of a supercategory."""
		all_imgs = []
		filter_int = self.coco.super_cat_ids[super_cat]
		filters = [self.coco.all_categories[c][0] for c in filter_int]

		for f in filters:
			img_paths, _ = self.coco.load_images_with_filter([f], suppress_print=suppress_print)
			all_imgs = all_imgs + img_paths
		all_imgs = list(set(all_imgs)) # get rid of duplicates
		return all_imgs

	def sample_pictures(self, images_labels, size, user_label, name, toggle_folder_creation, dump):
		"""Creates a sample of certain size from picture set.

		:param list images_labels: all images and labels we can sample from.
		:param int size: Size of the sample.
		:param str user_label: User label for the sample.
		:param str name: Name of the sample.
		:param bool toggle_folder_creation: Boolean flag toggling local copying of selected images rather than just
			dictionary representation.
		:param bool dump: Boolean flag indicating whether to dump the dictionary to json and write to disk.

		:return dictionary
		"""
		samples = random.sample(images_labels, size)
		return self._create_files(name, samples, user_label, toggle_folder_creation=toggle_folder_creation, dump=dump)

	def _create_files(self, name, samples, user_label, toggle_folder_creation, dump):
		"""Creates all necessary files given a sample.

		:param str name: Name of the sample.
		:param list samples: List of all samples.
		:param str user_label: User label of the sample.
		:param bool toggle_folder_creation: Toggles local image creation to a folder instead of just dictionary.
		:param bool dump: Toggles dumping of dictionary to json and write to disk.
		"""
		folder_path = self.data_path + name

		if os.path.isdir(folder_path) and toggle_folder_creation:
			# if directory exists, remove and create new fresh one
			shutil.rmtree(folder_path)
			time.sleep(1)
			os.mkdir(folder_path)
		elif toggle_folder_creation:
			# directory doesn't exist, create folder
			os.mkdir(folder_path)

		if dump:
			if not os.path.isdir(self.labels_path):
				# create folder if it doesnt exist already
				os.mkdir(self.labels_path)
			if os.path.isfile(self.labels_path + name + '.json'):
				os.remove(self.labels_path + name + '.json')
			labels_json = open(self.labels_path + name + '.json', 'w+')

		dst = self.data_path + name
		paths, ids, labels_all = [], [], []
		for img, labels in samples:
			img_id = self.coco.get_img_id_from_file(img)
			ids.append(img_id)
			paths.append(img)
			labels_all.append(labels)
			if toggle_folder_creation:
				# copy pictures to separate folders
				shutil.copy(self.coco.active_dir + img, dst + "/" + img)

		else:
			labels_dict = {'paths': paths, 'labels': labels_all, 'ids': ids, 'user_label': user_label}
		if dump:
			# dump the labels to json file
			json.dump(labels_dict, labels_json)
			labels_json.close()
		return labels_dict


