
<h1 align="center">VietSceneText_OCR</h1>

<!-- Status -->

<!-- <h4 align="center"> 
	🚧  VietSceneText_OCR 🚀 Under construction...  🚧
</h4> 

<hr> -->

<p align="center">
  <a href="#about">About</a> &#xa0; | &#xa0; 
  <a href="#starting">Starting</a> &#xa0;
</p>

<br>

## About ##

Framework end to end for SceneText OCR in wild

## Starting ##

<a href="https://drive.google.com/drive/folders/1DQzAcvsWsLo_dMgYSD4mHSPzH_gdkLEg?usp=sharing"> Download model here </a>

```bash
# Clone this project
$ git clone https://gitlab.com/tiendv/vietscenetext_framework.git

# Access
$ cd vietscenetext_framework

# Install dependencies
$ pip install -r requirements.txt

# download pretrain model
./download_model.sh

# unzip pretrain models
unzip models.zip
```
## Run the project ##

### Run detection on folder with file detect_text.py ###
```bash
# output will save at output_detection with txt format
python detect_text.py --run_on_folder True --folder_path path/to/folder/predict --config_method path/to/config --detection_weight path/to/detection
```
### Run detection on image with file detect_text.py ###
```bash
# output is list text bounding box of this image
python detect_text.py --run_on_folder False --image_path path/to/image/predict --config_method path/to/config --detection_weight path/to/detection
```
### Run recognition on folder with file recognize_text.py ###
```bash
# output will print in console with image name and text of this image
python recognize_text.py --run_on_folder True --folder_path path/to/predict/folder --config_method path/to/config --recognition_weight path/to/recognition/weight

```
### Run recognition with folder detection result (txt file) as input with file recognize_text.py ###
```bash
# output will json file and save at output_full_pipeline
python recognize_text.py --run_on_folder True --folder_path path/to/predict/folder --run_recognition_with_detection_result True --output_detection path/to/output/detection --config_method path/to/config --recognition_weight path/to/recognition/weight
```
### Run recognition on image ###
```bash
# output will print in console
python recognize_text.py --run_on_folder False --image_path path/to/predict/image
--config_method path/to/config --recognition_weight path/to/recognition/weight
```
### Using docker-compose file ###
```bash
# for gpu 
docker-compose -f docker-compose-gpu/yaml build
docker-compose -f docker-compose-gpu/yaml up

# for cpu
docker-compose -f docker-compose-cpu/yaml build
docker-compose -f docker-compose-cpu/yaml up
```
### Run end to end method: dict-guided with docker file
```bash
cd libs/DICT_GUIDED/docker

docker build -t dict_guided .

docker build --name <docker_name> --runtime nvidia -v <path/to/
vietscenetext_framework>:/<path/in/docker>/ -it dict_guided:latest /bin/bash

docker start <docker dict_guided>

docker attach <docker dict_guided>

cd to/path/vietscenetext/mapping/in/docker

conda activate base

<using command to run on folder or image of demo_dict_guided.py>

# visualize will auto save at visualize_output
# run on folder image
python demo_dict_guided.py --run_on_folder True --folder_path path/to/folder/predict --save_visualize True --opts MODEL.WEIGHTS path/to/model/weight 

# run on single image
python demo_dict_guided.py --run_on_folder False --image_path path/to/image/predict
--save_visualize True --opts MODEL.WEIGHTS path/to/model/weight
```

### Json file structure ###
```json
[
  {
    "image_name": name of image
    "result":
      {
        "x_min": x_min of bounding box
        "y_min" y_min of bounding box
        "x_max" x_max of bounding box
        "y_max" y_max of bounding box
        "text": text in this position
        
      }
      {
        "x_min": x_min of bounding box
        "y_min" y_min of bounding box
        "x_max" x_max of bounding box
        "y_max" y_max of bounding box
        "text": text in this position
        
      }
      }
      .
      .
      .
      {  
        "x_min": x_min of bounding box
        "y_min" y_min of bounding box
        "x_max" x_max of bounding box
        "y_max" y_max of bounding box
        "text": text in this position
      }
  }
  .
  .
  .
    {
    "image_name": name of image
    "result":
      {
        "x_min": x_min of bounding box
        "y_min" y_min of bounding box
        "x_max" x_max of bounding box
        "y_max" y_max of bounding box
        "text": text in this position
        
      }
      {
        "x_min": x_min of bounding box
        "y_min" y_min of bounding box
        "x_max" x_max of bounding box
        "y_max" y_max of bounding box
        "text": text in this position
        
      }
      }
      .
      .
      .
      {  
        "x_min": x_min of bounding box
        "y_min" y_min of bounding box
        "x_max" x_max of bounding box
        "y_max" y_max of bounding box
        "text": text in this position
      }
  }

]
```

<a href="#top">Back to top</a>
