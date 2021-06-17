"""
Load annotations of images and some other useful methods.
"""
import sys
from constants import coco_path
from sklearn.model_selection import train_test_split
from pycocotools.coco import COCO
import os, shutil, time
import json


class CocoDataTrain:
	"""Class with useful functions when handling the COCO API and a COCO annotated dataset."""

	def __init__(self, annotations='instances_train2017.json', data_dir=coco_path, dump=False,
				 json_dump_name='labels.json', active='train/'):
		self._data_dir = data_dir
		self._ann_dir = data_dir + 'annotations/'
		self.active_dir = data_dir + active
		self.coco = COCO(self._ann_dir + annotations)
		self.file_names = []
		self.labels = []

		self.category_keys = list(self.coco.catToImgs.keys())
		self.super_cat_ids, self.all_categories = self.create_id_structures()
		self.all_images = self.coco.getImgIds()  # get all image ids

		if dump:
			# dump json contents, very expensive operation, only execute when labels.json doesn't exist yet
			self._dump_labels(name=json_dump_name)
		with open(self._data_dir + json_dump_name, "r") as file:
			self.img_labels = json.load(file)  # load labels

		"""load all images into paths and labels list"""
		for image in self.all_images:
			# get string name and label of the image
			img_str = self.coco.loadImgs(image)[0]['file_name']
			label = self.img_labels[str(image)]

			# append paths and labels
			self.labels.append(label)
			self.file_names.append(img_str)

	def split_coco(self, test_size=20000, random_state=42, toggle_folder_creation=False, separate_batch=True):
		"""
		Function for splitting up training data into new train and test. The partitions are recorded in json format.

		:param separate_batch: toggles the creation of a completely separate batch of data, split off the test split,
			standard size is 5000
		"""
		print("Splitting up the training data.")
		# splits up train set (input and labels) into a test set of 20'000 and the train set of the remaining number
		trainImgs, testImgs, trainLabels, testLabels = train_test_split(self.file_names, self.labels,
																		  test_size=test_size,
																		  random_state=random_state)

		if separate_batch:
			# make another completely separate partition for results
			testImgs, batchImgs, testLabels, batchLabels = train_test_split(testImgs, testLabels,
																			test_size=5000, random_state=random_state)
			with open(self._data_dir + 'separate_batch.json', 'w+') as separate_batch_json:
				# creating batch folder and labels
				imgs_batch, labels_batch, img_ids_batch = [], [], []
				for img, label in zip(batchImgs, batchLabels):
					imgs_batch.append(img)
					labels_batch.append(label)
					img_id = self.get_img_id_from_file(img)
					img_ids_batch.append(img_id)
				# dumps the batch split to a json file
				json.dump({'paths': imgs_batch, 'labels': labels_batch, 'ids': img_ids_batch}, separate_batch_json)
				print("Batch with size:", len(batchImgs))
				print("Test split with size:", len(testImgs))
		print("Train split with size:", len(trainImgs))

		new_test_folder = self._data_dir + 'split_test/'
		new_train_folder = self._data_dir + 'split_train/'

		if toggle_folder_creation:
			if os.path.isdir(new_test_folder):
				shutil.rmtree(new_test_folder)
				os.remove(self._data_dir + 'split_test.json')
				# sleep as it sometimes gives an error when trying to create the directory too fast
				time.sleep(1)
			if os.path.isdir(new_train_folder):
				shutil.rmtree(new_train_folder)
				os.remove(self._data_dir + 'split_train.json')
				time.sleep(1)
			os.mkdir(new_train_folder)
			os.mkdir(new_test_folder)
		split_test_json = open(self._data_dir + 'split_test.json', 'w+')
		split_train_json = open(self._data_dir + 'split_train.json', 'w+')

		# creating test folder and labels
		imgs_test, labels_test, img_ids_test = [], [], []
		for img, label in zip(testImgs, testLabels):
			if toggle_folder_creation:
				shutil.copy(self.active_dir+img, new_test_folder + "/" + img)
			imgs_test.append(img)
			labels_test.append(label)
			img_id = self.get_img_id_from_file(img)
			img_ids_test.append(img_id)
		# dumps the test split to a json file
		json.dump({'paths': imgs_test, 'labels': labels_test, 'ids': img_ids_test}, split_test_json)

		# creating train folders and labels
		imgs_train, labels_train, img_ids_train = [], [], []
		for img, label in zip(trainImgs, trainLabels):
			if toggle_folder_creation:
				shutil.copy(self.active_dir+img, new_train_folder + "/" + img)
			imgs_train.append(img)
			labels_train.append(label)
			img_id = self.get_img_id_from_file(img)
			img_ids_train.append(img_id)
		# dumps the train split to a json file
		json.dump({'paths': imgs_train, 'labels': labels_train, 'ids': img_ids_train}, split_train_json)
		split_test_json.close()
		split_train_json.close()

	@staticmethod
	def get_img_id_from_file(file):
		"""Get the img id from a file name."""
		for i, nr in enumerate(file):
			if nr != '0':
				break
		id = int(file[i:-4])
		return id

	def load_images_with_filter(self, filter_list: list = [], suppress_print=False):
		"""Loads all images that satisfy the filter condition"""
		# all images containing category, see https://blog.roboflow.com/coco-dataset/ for category list
		# or https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoDemo.ipynb
		# also see supercategories
		category_id = self.coco.getCatIds(catNms=filter_list)

		imgIds = self.coco.getImgIds(catIds=category_id)
		if not suppress_print:
			print(len(imgIds), "Pictures with class:", filter_list)
		img_paths_filter, labels_filter = [], []
		for im in imgIds:
			img_str = self.coco.loadImgs(im)[0]['file_name']
			label = self.img_labels[str(im)]

			img_paths_filter.append(img_str)
			labels_filter.append(label)

		return img_paths_filter, labels_filter

	def load_split_imgs(self, name, filter_list: list = [], path=coco_path):
		"""Loads all images that satisfy the filter condition but only for the split partitions."""
		try:
			with open(path+name+'.json', "r") as file:
				split_dict = json.load(file)
		except Exception as e:
			print("Couldn't load the split json file, naming or path is most likely wrong.")
			print(name, "at", path)
			sys.exit()
		# keys: paths, labels, ids
		all_labels = split_dict['labels']
		all_imgs = split_dict['paths']

		category_id = self.coco.getCatIds(catNms=filter_list)

		img_paths_filter, labels_filter = [], []
		# I only want the labels from category_id
		for i, img_ls in enumerate(all_labels):
			for l in img_ls:
				if l in category_id:
					# if one of the single labels is in the categories we want, add it to the list,
					# then we break so that we do not have duplicates
					labels_filter.append(img_ls)
					img_paths_filter.append(all_imgs[i])
					break
		return img_paths_filter, labels_filter

	def create_id_structures(self):
		"""Function that creates two dictionaries that allow easy retrieval of id of label in respect to name
		and all ids belonging to a certain supercategory. """
		all_categories = {}
		super_cat_ids = {}
		ids = self.coco.dataset['categories']
		for e in ids:
			cat_id = e['id']
			cat_name = e['name']
			super_cat = e['supercategory']
			all_categories[cat_id] = (cat_name, super_cat)
			if super_cat in super_cat_ids:
				# if super-category key was already created, add new id to list
				super_cat_ids[super_cat] += [cat_id]
			else:
				# if not created, set value of that key to a list with that id
				super_cat_ids[super_cat] = [cat_id]

		return super_cat_ids, all_categories

	def return_image_classes(self, image_id):
		"""Given an image, it returns all categories / classes present in the picture."""
		cats_of_images = []
		for cat_key in self.category_keys:
			imgs = self.coco.catToImgs[cat_key]
			if image_id in imgs:
				cats_of_images.append(cat_key)

		return cats_of_images

	def cat_to_super_cat(self, categories: list):
		"""Translates normal categories to it's supercategories."""
		lst = []
		for c in categories:
			sc = self.all_categories[c][1]
			lst.append(sc)
		return lst

	def _dump_labels(self, name):
		"""Function that dumps to json file.
		It contains a dictionary with every image_id of the train dataset as a key and value the respective
		labels belonging to that image. There are however no duplicate labels, even when the same object
		appears in the scene multiple times.
		"""
		print("Dumping to json:", name)
		image_labels = {}
		for img_id in self.all_images:
			cats_of_images = self.return_image_classes(img_id)
			image_labels[img_id] = cats_of_images
		with open(self._data_dir + name, "w+") as f:
			json.dump(image_labels, f)
		print("Finished dumping labels")


class CocoDataVal(CocoDataTrain):
	"""Class inheriting from CocoDataFunctions only for validation data. """

	def __init__(self, annotations='instances_val2017.json', json_dump_name='labels_val.json', dump=False):
		super().__init__(annotations=annotations, json_dump_name=json_dump_name, active='val/', dump=dump)

