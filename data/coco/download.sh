# simple shell script downloading the training, validation and annotations of COCO 2017
# only really works on linux

wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip

unzip train2017.zip
unzip val2017.zip

mv train2017 train
mv val2017 val

rm train2017.zip
rm val2017.zip

wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip

rm annotations_trainval2017.zip

