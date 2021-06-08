
COCO_INSTANCE_CATEGORY_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


COCO_SUPER_CATEGORIES = ['outdoor', 'food', 'indoor', 'appliance', 'sports', 'animal', 'vehicle', 'furniture',
                'accessory', 'electronic', 'kitchen']

# path options for the local implementation, change accordingly
data_path = '../data/'
coco_path = data_path + 'coco/'
train_path = coco_path + 'train/'
val_path = coco_path + 'val/'
model_path = '../models/'
sub_sets = data_path + 'sub_sets/'

"""Person is omitted in our research as it can be found in over half of the images in the training set, outshadowing 
every other class by a lot. Therefore to make the data more balanced we ignore it completely. """

# still include person
COCO_SUPER_CATEGORIES_WITH_PERSON = ['outdoor', 'food', 'indoor', 'appliance', 'sports', 'person', 'animal', 'vehicle', 'furniture',
                'accessory', 'electronic', 'kitchen']