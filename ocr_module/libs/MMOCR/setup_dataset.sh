#train textdetion
mkdir data
cd data
gdown --id 1Q9TuvZTOYF4GRciLddVdpYyWXpJQfRjl
unzip data_pan_14_points.zip 
mkdir annotations
mkdir imgs
mv data_pan/train/img imgs/training
mv data_pan/val/img imgs/test
mv data_pan/train/label annotations/training
mv data_pan/val/label annotations/test
rm -rf data_pan
rm -rf data_pan_14_points.zip
cd ..

python3 tools/data/textdet/ctw1500_converter.py data -o data --split-list training test