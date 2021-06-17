"""
Script that creates training, test and validation data for training a classifier on user level.
Works on the coco training dataset. The split function should also be called from here if needed.
"""
import shutil

from coco_data_funcs import CocoDataTrain, CocoDataVal
from create_user import UserSampling
from tqdm import tqdm
import time, os
from constants import COCO_SUPER_CATEGORIES, data_path, sub_sets


def create_dedicated_test_category(coco, category, path=data_path + 'sub_sets/special_testing_data/', nr_of_users=100,
                                   img_per_user=20, super_cat=True, toggle_folder_creation=False, split=True,
                                   split_name="split_test"):
    """Creates special testing data. See create_all_special_test_sets()."""
    path = path if path[-1] == '/' else path + '/'

    flush(path)
    users = UserSampling(coco=coco, labels_path=path)
    since = time.time()

    users_list = []

    for i in range(nr_of_users):
        dict_user = users.create_user(name="user_" + category + '_' + str(i), category=category, super_cat=super_cat,
                                      size=img_per_user,
                                      toggle_folder_creation=toggle_folder_creation, dump=True, suppress_print=True,
                                      split=split, split_name=split_name)
        users_list.append(dict_user)

    time_elapsed = time.time() - since
    print('Creation of {} data complete in {:.0f}m {:.0f}s'.format(category,
        time_elapsed // 60, time_elapsed % 60))
    return users_list


def create_all_special_test_sets(val=False):
    """Wrapper for creating special test sets to test the framework on. """
    if val:
        for sc in COCO_SUPER_CATEGORIES:
            create_dedicated_test_category(cocoVal, sc,
                                           path=data_path + 'sub_sets/special_testing_data/'+sc+'_val/', nr_of_users=20,
                                           split=False, split_name="")
    else:
        for sc in COCO_SUPER_CATEGORIES:
            create_dedicated_test_category(cocoTrain, sc,
                                           path=data_path + 'sub_sets/special_testing_data/'+sc+'_test/')


def create_user_dataset(coco, path=data_path+'sub_sets/split_train_test_val_data/', nr_of_users=10, img_per_user=20, super_cat=True,
                        toggle_folder_creation=False, split=False, split_name=""):
    """Generic function for creating train, validation and test data used for training a custom classifier for
    faster RCNN. It's data on user level. Does not balance data for a binary classifier. Every class is represented for
    1 / n, with n being the amount of categories. Not used in paper.
    """
    path = path if path[-1] == '/' else path + '/'

    flush(path[:-1])
    users = UserSampling(coco=coco, imgs_path=path, labels_path=path[:-1] + '_labels/')
    since = time.time()

    users_list = []

    for sc in tqdm(COCO_SUPER_CATEGORIES):
        for i in range(nr_of_users):
            dict_user = users.create_user(name="user_" + sc + '_' + str(i), category=sc, super_cat=super_cat,
                                          size=img_per_user,
                                          toggle_folder_creation=toggle_folder_creation, dump=True, suppress_print=True,
                                          split=split, split_name=split_name)
            users_list.append(dict_user)

    time_elapsed = time.time() - since
    print('Creation of data complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    return users_list


def flush(path):
    """Removes old data."""
    if os.path.isdir(path):
        shutil.rmtree(path)


def custom_data(coco, target, path=data_path+'/sub_sets/split_train_test_val_data/', nr_of_users=2, img_per_user=80,
                super_cat=True, toggle_folder_creation=False, split=False, split_name=""):
    """Creates data specifically for training a net to return binary prediction for a category. Target is the category
    we're trying to train the network on. The data will subsequently be half / half,
    positive match target / negative match target.
    """
    # create training data with more vehicles in it
    path = path if path[-1] == '/' else path + '/'

    flush(path)
    users = UserSampling(coco=coco, labels_path=path)
    since = time.time()

    users_list = []

    for sc in tqdm(COCO_SUPER_CATEGORIES, desc=target):
        if sc == target:
            for i in range(nr_of_users*(len(COCO_SUPER_CATEGORIES)-1)):
                dict_user = users.create_user(name="user_" + sc + '_' + str(i), category=sc, super_cat=super_cat,
                                              size=img_per_user,
                                              toggle_folder_creation=toggle_folder_creation, dump=True, suppress_print=True,
                                              split=split, split_name=split_name)
                users_list.append(dict_user)
        else:
            for i in range(nr_of_users):
                dict_user = users.create_user(name="user_" + sc + '_' + str(i), category=sc, super_cat=super_cat,
                                              size=img_per_user,
                                              toggle_folder_creation=toggle_folder_creation, dump=True, suppress_print=True,
                                              split=split, split_name=split_name, anti_category=target, super_cat_anti=True)
                users_list.append(dict_user)

    time_elapsed = time.time() - since
    print('Creation of data complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    return users_list


if __name__ == '__main__':
    cocoTrain = CocoDataTrain()
    cocoVal = CocoDataVal()
    # cocoTrain.split_coco() # if you want to make a new split, uncomment this, otherwise unpack global_split.zip

    # category = 'animal'
    for sc in COCO_SUPER_CATEGORIES:

        # train data
        custom_data(cocoTrain, sc, nr_of_users=8, img_per_user=80,
                    path=sub_sets + 'split_train_test_val_data/' + sc + '_train_data/', split=True,
                    split_name='split_train')
        # test data
        custom_data(cocoTrain, sc, nr_of_users=2, img_per_user=20,
                    path=sub_sets+'split_train_test_val_data/' + sc + '_test_data/', split=True,
                    split_name='split_test')
        # val data
        custom_data(cocoVal, sc, nr_of_users=2, img_per_user=80,
                    path=sub_sets + 'split_train_test_val_data/' + sc + '_val_data/')



