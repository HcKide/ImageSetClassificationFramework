import numpy as np
from coco_data_funcs import CocoDataVal, CocoDataTrain
from constants import COCO_INSTANCE_CATEGORY_NAMES, COCO_SUPER_CATEGORIES
import matplotlib.pyplot as plt


def counts_modified(target_sc, coco):
	"""Counts and plots all the instances, highlights the instances of a super-category with red."""
	amounts = []
	names = []

	target_cs = coco.super_cat_ids[target_sc]
	all_cats = []
	for sc in COCO_SUPER_CATEGORIES:
		cats = coco.super_cat_ids[sc]
		all_cats += cats
		for i, c in enumerate(cats):
			imgIds = coco.coco.getImgIds(catIds=c)
			amounts.append(len(imgIds))
			names.append(COCO_INSTANCE_CATEGORY_NAMES[c - 1])

	filtered_instances = []
	for c in COCO_INSTANCE_CATEGORY_NAMES:
		if c != 'N/A':
			filtered_instances.append(c)

	plt.figure(figsize=(12, 10))
	plt.grid(zorder=0)

	selection = ['orange', 'green', 'blue']
	special = 'red'
	cols = [] # ugly fix to remove the shift of colours
	for i, ci in enumerate(all_cats):
		if ci in target_cs:
			cols.append(special)
		else:
			cols.append(selection[i % 3])

	plt.bar(names, amounts, color=cols, zorder=3)
	plt.xlim(-1, len(filtered_instances[1:]))
	plt.xticks(names, names, rotation=90)
	plt.xlabel('Categories')
	plt.ylabel('Counts')
	plt.show()


def counts_all(coco):
	"""Counts and plots all occurences of normal instances. """
	amounts = []
	names = []

	for sc in COCO_SUPER_CATEGORIES:
		cats = coco.super_cat_ids[sc]

		for i, c in enumerate(cats):
			imgIds = coco.coco.getImgIds(catIds=c)
			# print(COCO_INSTANCE_CATEGORY_NAMES[c], ":", len(imgIds))
			amounts.append(len(imgIds))
			names.append(COCO_INSTANCE_CATEGORY_NAMES[c-1])

	filtered_instances = []
	for c in COCO_INSTANCE_CATEGORY_NAMES:
		if c != 'N/A':
			filtered_instances.append(c)

	plt.figure(figsize=(12, 10))
	plt.grid(zorder=0)
	cols = ['orange', 'green', 'blue']
	plt.bar(names, amounts, color=cols, zorder=3)
	plt.xlim(-1, len(filtered_instances[1:]))
	plt.xticks(np.arange(len(filtered_instances[1:])), filtered_instances[1:], rotation=90)
	plt.xlabel('Categories')
	plt.ylabel('Counts')
	plt.show()


def counts_per_supercat(supercat, coco, show=True):
	"""Counts and plots all occurences of the instances of a super-category."""
	cats = coco.super_cat_ids[supercat]
	print(cats)

	amounts = []
	names = []
	for i, c in enumerate(cats):
		imgIds = coco.coco.getImgIds(catIds=c)
		print(COCO_INSTANCE_CATEGORY_NAMES[c-1], ":", len(imgIds))
		amounts.append(len(imgIds))
		names.append(COCO_INSTANCE_CATEGORY_NAMES[c-1])

	if show:
		cols = ['orange', 'green', 'blue']
		plt.figure(figsize=(12, 10))
		plt.grid(zorder=0)
		plt.bar(names, amounts, zorder=3, color=cols)
		plt.xticks(names, names, rotation=90)
		# plt.xlabel('Categories')
		plt.ylabel('Occurences')
		plt.show()


if __name__ == '__main__':
	# coco = CocoDataVal()
	coco = CocoDataTrain()

	counts_per_supercat('furniture', coco=coco, show=True)
	counts_per_supercat('sports', coco=coco, show=True)
	# counts_all(coco=cTrain)
	# counts_modified('sports', coco=coco)
