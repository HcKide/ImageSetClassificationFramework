## Image Set Classification Framework
Public Repository for my Bachelor Thesis in Artificial Intelligence, 
Stijn Boosman

Radboud University

#### Understanding image set classifiers for future evaluation of adversarial profiles to gain control of our own privacy.

Important! Before using this project there a few things to note:

- The COCO dataset must be downloaded and placed in the appropriate folders, use the shellscript in coco/ for easy retrieval.
- In the constants.py file, the data paths need to be edited to one's preferences if the data is located somewhere else. 

Link to VGG models: (to appear)

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
│   │   ├── split_test.json     <- (Optional) Test partition of a train/test split when training. Found in global_split-zip.
│   │   └── split_train.json    <- (Optional) Train partition of a train/test split when training. Found in global_split.zip.
│   └── sub_sets            
│       ├── all_categories.txt  <- All categories present in the COCO dataset, although see constants.py for categories that have been deleted (N/A)
│       ├── thresholds          <- Folder containing all the thresholds used to compute tables in experimental results. The tresholds were optimized for highest accuracy.
│       └── split_train_test_val_data    <- Folder with train, test and val data of user-labelled sets of images.  
├── models                      <- Pretrained model and dict states. See drive link to download models. 
├── requirements.txt   
├── docs                        <- Folder containing my Bsc Thesis Presentation and the Thesis paper itself.                   
└── src                         <- Source code                
    ├── __init__.py 
    ├── constants.py            <- Holds constants like a list of all coco categories and supercategories, used throughout the whole project.
    └── ....                    <- All other source code files

```