# ThesisCode
Repository for my Bachelor Thesis, 
Stijn Boosman

Radboud University

#### Understanding image set classifiers for future evaluation of adversarial profiles to gain control of our own privacy.

Important! Before using this project there a few things to note:

- The COCO dataset must be downloaded and placed in the appropriate folders
- In the constants.py file, the data paths need to be edited to one's preferences if the data is located somewhere else

Project structure:
```
├── README.md          
├── data
│   ├── coco
│   │   ├── annotations         <- Train/Val Annotations folder containing the official instances of the MS COCO dataset (2017)
│   │   ├── train               <- Training images from the MS Coco Dataset (2017), download here: https://cocodataset.org/#download
│   │   ├── val                 <- Validation Images from the MS Coco Datset (2017)
│   │   ├── labels.json         <- Custom generated json file, with key, value pair as follows: "img id" : labels of that image, for training only
│   │   ├── labels_val.json     <- Custom generated json file, for validation only
│   │   ├── split_test.json     <- (Optional) Test partition of a train/test split when training. 
│   │   └── split_train.json    <- (Optional) Train partition of a train/test split when training.
│   └── sub_sets            
│       ├── all_categories.txt  <- All categories present in the COCO dataset, although see constants.py for categories that have been deleted (N/A)
│       └── artificial_users    <- Folder with different types of artificial users, all sampled from validation data. Used to test performances.
├── models                      <- Pretrained model and dict states.
├── requirements.txt            
├── setup.py           
└── src                         <- Source code                
    ├── __init__.py 
    ├── constants.py            <- Holds constants like a list of all coco categories and supercategories, used throughout the whole project.
    └── ....                    <- All other source code files

```