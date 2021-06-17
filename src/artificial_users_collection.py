"""Collection of classes and function to create artificial users."""

from create_user import UserSampling
from coco_data_funcs import CocoDataVal, CocoDataTrain
from constants import COCO_SUPER_CATEGORIES, COCO_INSTANCE_CATEGORY_NAMES, sub_sets, train_path
import time, copy, cv2, os
from tqdm import tqdm
import shutil
import matplotlib.pyplot as plt
from utils import merge_dict, merge_folders, dump_json


class Users:
	"""Users class, containing high-level functions for creating different users with special parameters.
	With these functions we can create users that vary along our main two aspects, dimension and distribution.
	"""

	def __init__(self, labels_path=None, userSampling=None):
		if userSampling is not None:
			self.userSampling = userSampling
		else:
			cocoV = CocoDataVal()
			if labels_path:
				self.userSampling = UserSampling(cocoV, labels_path=labels_path)
			else:
				self.userSampling = UserSampling(cocoV)

	def super_cat_users(self, size=20, dump=True):
		"""Create a simple, randomly sampled user for every super category"""
		super_cat_users_dict = {}
		for sc in COCO_SUPER_CATEGORIES:
			dict_user = self.userSampling.create_user(name="user_"+sc, category=sc, super_cat=True, size=size,
													  dump=dump, suppress_print=True)
			super_cat_users_dict[sc] = dict_user
		return super_cat_users_dict

	def vary_img_p_user(self, sizes=[10, 20, 30, 40, 50], path=sub_sets+'artificial_users/varying_size_users/', dump=True):
		"""Creates multiple users varying in image set size. """

		for sc in COCO_SUPER_CATEGORIES:
			users_per_cat = {}
			for s in sizes:
				dict_user = self.userSampling.create_user(name="user_"+sc, category=sc, super_cat=True, size=s,
														  dump=False, suppress_print=False, split=True, split_name='split_test')
				users_per_cat[s] = dict_user

			if dump:
				dump_json(users_per_cat, 'users_'+sc, path)

	def add_negatives(self, anti_category, r, super_cat, size, toggle_folder_creation, split=False, split_name=""):
		"""Creates a negative user, avoiding the given target parameter."""
		idx = COCO_SUPER_CATEGORIES.index(anti_category)
		targets = COCO_SUPER_CATEGORIES[:idx]+COCO_SUPER_CATEGORIES[idx+1:]

		dict_user_neg = self.userSampling.create_user(name="user_non_" + anti_category + "_p0_r" + str(r) + '_t',
		                                              category=targets, super_cat=True, size=size, dump=False,
		                                              suppress_print=True,
		                                              anti_category=anti_category, super_cat_anti=super_cat,
		                                              custom_user_label="not-" + anti_category,
		                                              toggle_folder_creation=toggle_folder_creation,
		                                              split=split, split_name=split_name
		                                              )
		return dict_user_neg

	def apply_data_recipe(self, category, path, super_cat=False, repeat=1, img_per_user=20,
	                      percentages=[10, 60, 90, 100], dump=True, toggle_folder_creation=False, val=False,
	                      single_object=None, negative=False):
		"""Applies the recipe for varying distributions of a specified category."""
		# create x amount of users varying in distributions
		users = {p: [] for p in percentages}

		if single_object:
			dump_str = single_object
			target = single_object
			# all non-targets within same superclass
			lst = self.userSampling.coco.super_cat_ids[category]
			anti_object = []
			for c in lst:
				anti_object.append(self.userSampling.coco.all_categories[c][0])
			idx = anti_object.index(target)
			del anti_object[idx]
			super_cat = False
		else:
			dump_str = category
			target = category
			anti_object = None

		if not val:
			for r in tqdm(range(repeat), desc=category + " data recipe"):
				for p in percentages:
					# rounds down if float occurs
					size_target = int(p/100 * img_per_user)
					size_filler = img_per_user - size_target
					dict_user_target = self.userSampling.create_user(name="user_"+target+'_p'+str(p)+'_r'+str(r)+'_t', category=target,
																		 super_cat=super_cat, size=size_target, dump=False, suppress_print=True,
																	     custom_user_label=category,
																	     toggle_folder_creation=toggle_folder_creation,
																		 split=True, split_name='split_test', anti_category=anti_object, super_cat_anti=False
																	 )
					filler_cats = self._create_filler_categories(exclusion=category, super_cat=True)
					dict_user_filler = self.userSampling.create_user(name="user_"+target+'_p'+str(p)+'_r'+str(r)+'_f', category=filler_cats,
																		 super_cat=False, size=size_filler, dump=False, suppress_print=True,
																		 anti_category=category, super_cat_anti=True,
																	     custom_user_label=category,
																	     toggle_folder_creation=toggle_folder_creation,
																		 split=True, split_name='split_test'
																	 )
					dict_user_merged = merge_dict(dict_user_target, dict_user_filler)
					users[p] = users[p] + [dict_user_merged]
					if negative:
						neg_dct = self.add_negatives(category, r, True, img_per_user, toggle_folder_creation, split=True,
						                             split_name='split_test')
						users[p] = users[p] + [neg_dct]
		else:
			# val should only be activated with the cocoVal class otherwise it's useless
			for r in tqdm(range(repeat), desc=category + " data recipe"):
				for p in percentages:
					# rounds down if float occurs
					size_target = int(p/100 * img_per_user)
					size_filler = img_per_user - size_target
					dict_user_target = self.userSampling.create_user(name="user_"+target+'_p'+str(p)+'_r'+str(r)+'_t', category=target,
																		 super_cat=super_cat, size=size_target, dump=False, suppress_print=True,
																	     custom_user_label=category,
																	     toggle_folder_creation=toggle_folder_creation,
					                                                 anti_category=anti_object, super_cat_anti=False
																	 )
					filler_cats = self._create_filler_categories(exclusion=category, super_cat=True)
					dict_user_filler = self.userSampling.create_user(name="user_"+target+'_p'+str(p)+'_r'+str(r)+'_f', category=filler_cats,
																		 super_cat=False, size=size_filler, dump=False, suppress_print=True,
																		 anti_category=category, super_cat_anti=True,
																	     custom_user_label=category,
																	     toggle_folder_creation=toggle_folder_creation
																	 )
					dict_user_merged = merge_dict(dict_user_target, dict_user_filler)
					users[p] = users[p] + [dict_user_merged]
					if negative:
						neg_dct = self.add_negatives(category, r, True, img_per_user, toggle_folder_creation)
						users[p] = users[p] + [neg_dct]

		if dump:
			dump_json(users, name='users_'+dump_str, path=path)
		if toggle_folder_creation:
			merge_folders(path+'imgs/', dump_str, percentages, repeat)
		return users

	def _create_filler_categories(self, exclusion: str or list, super_cat=False):
		"""Creates categories for filler images which will exclude the target category/-ies."""
		if super_cat:
			if type(exclusion) != list:
				exclusion = [exclusion]
			exclusion_int = []
			for e in exclusion:
				exclusion_int += self.userSampling.coco.super_cat_ids[e]
			exclusions = [self.userSampling.coco.all_categories[c][0] for c in exclusion_int]
			exclusions = list(set(exclusions))
		else:
			if type(exclusion) is list:
				exclusions = exclusion
			else:
				exclusions = [exclusion]
		cats = copy.copy(COCO_INSTANCE_CATEGORY_NAMES)
		for c in exclusions:
			idx = cats.index(c)
			del cats[idx]
		# remove duplicates
		cats = list(set(cats))
		try:
			# delete N/A entry
			del cats[cats.index('N/A')]
		except:
			pass
		return cats

	def recipe_all_supercats(self, path=sub_sets+'/artificial_users/recipe_users/', toggle_folder_creation=False,
	                         repeat=1):
		"""Creates a bunch of sampled users using the data recipe for all supercategories."""
		if toggle_folder_creation:
			if os.path.isdir(path+'imgs/'):
				shutil.rmtree(path+'imgs/')
				time.sleep(1)
			os.mkdir(path+'imgs/')
			self.userSampling.data_path = path + 'imgs/'

		all_users = {}
		for sc in COCO_SUPER_CATEGORIES:
			users = self.apply_data_recipe(sc, path, super_cat=True, repeat=repeat, img_per_user=20,
			                               toggle_folder_creation=toggle_folder_creation)
			all_users[sc] = users
		return all_users

	def _display_bbox(self, dict_user):
		"""Display bounding boxes for an image."""

		files = dict_user['paths']
		img_ids = dict_user['ids']
		labels = dict_user['labels']
		for i, f in enumerate(files):
			f_path = train_path + f
			img = cv2.imread(f_path)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			img_id = img_ids[i]
			coco = self.userSampling.coco.coco
			annIds = coco.getAnnIds(imgIds=img_id, catIds=labels[i], iscrowd=None)
			anns = coco.loadAnns(annIds)
			for j, a in enumerate(anns):
				[x, y, w, h] = a['bbox']
				xy = (int(x), int(y))
				wh = (int(x + w), int(y + h))
				cv2.rectangle(img, xy, wh, (0, 255, 0), 1)
				cat_id = a['category_id']
				l = COCO_INSTANCE_CATEGORY_NAMES[cat_id - 1]
				cv2.putText(img, l, xy, cv2.FONT_HERSHEY_COMPLEX_SMALL,
				            1, (0, 255, 0), thickness=1)

			plt.figure(figsize=(8, 12))
			plt.imshow(img)
			plt.xticks([])
			plt.yticks([])
			plt.show()

	def get_one_image(self, category, path=sub_sets + 'artificial_users/', dump=False, super_cat=True,
	                  toggle_folder_creation=False, display=True):
		"""Function not used."""
		if toggle_folder_creation:
			if os.path.isdir(path+'imgs/'):
				shutil.rmtree(path+'imgs/')
				time.sleep(1)
			os.mkdir(path+'imgs/')
			self.userSampling.data_path = path + 'img/'
		dict_user = self.userSampling.create_user(name="user_"+category, category=category, super_cat=super_cat, dump=False,
		                                          split=True, split_name='split_test', suppress_print=False, size=1)
		print(dict_user)
		if display:
			self._display_bbox(dict_user)

		if dump:
			dump_json(dict_user, 'users_'+category, path)


def main(generic=False, recipe=False, vary_size=False, dimension=False, threshold10p=False, threshold60p=False,
         threshold90p=False, target=""):
	"""Main function with different flags.

	:param generic: Flag for creating generic users, not used in any results in the paper. Mainly for testing purposes.
	:param recipe: Flag for creating the old way of recipe users. Deprecated.
	:param vary_size: Flag for creating users with differing sizes, i.e. 10 images, 20 images etc.
	:param dimension: Flag for creating users along a target object dimension. Change the target variable for that flag
		accordingly. The paper uses 'toilet' and 'snowboard'.
	:param threshold10p: Flag for creating users with 10 percent target images.
	:param threshold60p: Flag for creating users with 60 percent target images.
	:param threshold90p: Flag for creating users with 90 percent target images.
	:param target: Target for dimension flag.
	"""
	since = time.time()
	coco_T = CocoDataTrain()
	coco_V = CocoDataVal()
	userS = UserSampling(coco_T, imgs_path=sub_sets + '/artificial_users/recipe_users/imgs/',
	                     labels_path=sub_sets + '/artificial_users/user_sc_labels/')
	users = Users(userSampling=userS)
	if generic:
		super_cats_dict = users.super_cat_users()
	if recipe:
		users.recipe_all_supercats(path=sub_sets + '/artificial_users/recipe_users_more/',
								   toggle_folder_creation=False, repeat=100)
	if vary_size:
		users.vary_img_p_user()

	if dimension:
		u_label = users.userSampling.coco.all_categories[COCO_INSTANCE_CATEGORY_NAMES.index(target) + 1][1]
		for percentage in [10, 60, 90]:

			users.userSampling.data_path = sub_sets + 'threshold_users/'+str(percentage)+'p_'+target+'/test/imgs/'
			users.apply_data_recipe(u_label, path=sub_sets + 'threshold_users/'+str(percentage)+'p_'+target+'/test/',
			                        percentages=[percentage], repeat=50, single_object=target, negative=True)
			# val data
			userV = UserSampling(coco_V, imgs_path=sub_sets + 'threshold_users/'+str(percentage)+'p_'+target+'/val/imgs/',
			                     labels_path=sub_sets + 'threshold_users/'+str(percentage)+'p_'+target+'/val/')
			usersV = Users(userSampling=userV)
			usersV.apply_data_recipe(u_label, path=sub_sets + 'threshold_users/'+str(percentage)+'p_'+target+'/val/',
			                         percentages=[percentage], repeat=10, val=True, single_object=target, negative=True)

	target = 'sports'  # change accordingly for p-users
	if threshold10p:
		# test data
		users.userSampling.data_path = sub_sets + 'threshold_users/10p/test/imgs/'
		users.apply_data_recipe(target, path=sub_sets+'threshold_users/10/test/', super_cat=True, percentages=[10],
		                        repeat=50, negative=True)
		# val data
		userV = UserSampling(coco_V, imgs_path=sub_sets + 'threshold_users/10p/val/imgs/',
	                     labels_path=sub_sets + 'artificial_users/user_sc_labels/')
		usersV = Users(userSampling=userV)
		usersV.apply_data_recipe(target, path=sub_sets+'threshold_users/10p/val/', super_cat=True, percentages=[10],
								repeat=10, val=True, negative=True)

	if threshold60p:
		# test data
		users.userSampling.data_path = sub_sets + 'threshold_users/60p/test/imgs/'
		users.apply_data_recipe(target, path=sub_sets + 'threshold_users/60p/test/', super_cat=True, percentages=[60],
		                        repeat=50, negative=True)
		# val data
		userV = UserSampling(coco_V, imgs_path=sub_sets + 'threshold_users/60p/val/imgs/',
		                     labels_path=sub_sets + 'artificial_users/user_sc_labels/')
		usersV = Users(userSampling=userV)
		usersV.apply_data_recipe(target, path=sub_sets + 'threshold_users/60p/val/', super_cat=True, percentages=[60],
		                         repeat=10, val=True, negative=True)

	if threshold90p:
		# test data
		users.userSampling.data_path = sub_sets + 'threshold_users/90p/test/imgs/'
		users.apply_data_recipe(target, path=sub_sets + 'threshold_users/90p/test/', super_cat=True, percentages=[90],
		                        repeat=50, negative=True)
		# val data
		userV = UserSampling(coco_V, imgs_path=sub_sets + 'threshold_users/90p/val/imgs/',
		                     labels_path=sub_sets + 'artificial_users/user_sc_labels/')
		usersV = Users(userSampling=userV)
		usersV.apply_data_recipe(target, path=sub_sets + 'threshold_users/90p/val/', super_cat=True, percentages=[90],
		                         repeat=10, val=True, negative=True)

	time_elapsed = time.time() - since
	print('Creation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


if __name__ == '__main__':
	# set correct flags
	main(generic=False, recipe=False, vary_size=False, dimension=True, threshold90p=False, threshold60p=False,
	     threshold10p=False, target='toilet')
	main(generic=False, recipe=False, vary_size=False, dimension=True, threshold90p=False, threshold60p=False,
	     threshold10p=False, target='snowboard')
