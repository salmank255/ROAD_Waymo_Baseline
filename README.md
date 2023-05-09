# ROAD-Waymo Baseline for ROAD, ROAD-Waymo, ROAD++ and UCF-24 dataset
This repository contains code for 3D-RetinaNet, a novel Single-Stage action detection newtwork proposed along with [ROAD-Waymo dataset](https://github.com/salmank255/Road-waymo-dataset) and [ROAD dataset](https://github.com/gurkirt/road-dataset). This code contains training and evaluation for ROAD-Waymo, ROAD and UCF-24 datasets. 



## Table of Contents
- <a href='#requirements'>Requirements</a>
- <a href='#training-3d-retinanet'>Training 3D-RetinaNet</a>
- <a href='#testing-and-building-tubes'>Testing and Building Tubes</a>
- <a href='#performance'>Performance</a>
- <a href='#todo'>TODO</a>
- <a href='#citation'>Citation</a>
- <a href='#references'>Reference</a>


## Requirements
We need three things to get started with training: datasets, kinetics pre-trained weight, and pytorch with torchvision and tensoboardX. 

### Dataset download an pre-process

- We currently support the following three dataset.
    - [ROAD-Waymo dataset](https://github.com/salmank255/Road-waymo-dataset)
    - [ROAD dataset](https://github.com/gurkirt/road-dataset) in dataset release [paper](https://arxiv.org/pdf/2102.11585.pdf)
    - [UCF24](http://www.thumos.info/download.html) with [revised annotations](https://github.com/gurkirt/corrected-UCF101-Annots) released with our [ICCV-2017 paper](https://arxiv.org/pdf/1611.08563.pdf).

- Visit [ROAD-Waymo dataset](https://github.com/salmank255/Road-waymo-dataset) for download and pre-processing. 
- Visit [ROAD dataset](https://github.com/gurkirt/road-dataset) for download and pre-processing. 


### Pytorch and weights

  - Install [Pytorch](https://pytorch.org/) and [torchvision](http://pytorch.org/docs/torchvision/datasets.html)
  - INstall tensorboardX viad `pip install tensorboardx`
  - Pre-trained weight on [kinetics-400](https://deepmind.com/research/open-source/kinetics). Download them by changing current directory to `kinetics-pt` and run the bash file [get_kinetics_weights.sh](./kinetics-pt/get_kinetics_weights.sh). OR Download them from  [Google-Drive](https://drive.google.com/drive/folders/1xERCC1wa1pgcDtrZxPgDKteIQLkLByPS?usp=sharing). Name the folder `kinetics-pt`, it is important to name it right. 



## Training 3D-RetinaNet
- We assume that you have downloaded and put dataset and pre-trained weight in correct places.    
- To train 3D-RetinaNet using the training script simply specify the parameters listed in `main.py` as a flag or manually change them.

Let's assume that you extracted dataset in `/home/user/road-waymo/` and weights in `/home/user/kinetics-pt/` directory then your train command from the root directory of this repo is going to be:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py /home/user/ /home/user/  /home/user/kinetics-pt/ --MODE=train --ARCH=resnet50 --MODEL_TYPE=I3D --DATASET=road_waymo --TEST_DATASET=road_waymo --TRAIN_SUBSETS=train --VAL_SUBSETS=val --SEQ_LEN=8 --TEST_SEQ_LEN=8 --BATCH_SIZE=4 --LR=0.0041
```

Second instance of `/home/user/` in above command specifies where checkpoint weight and logs are going to be stored. In this case, checkpoints and logs will be in `/home/user/road-waymo/cache/<experiment-name>/`.
```
--ARCH          ---> By default it's resent50 but our code also support resnet101
--MODEL_TYPE    ---> We support six different models including I3D and SlowFast
--DATASET       ---> Dataset specifiy the training dataset as we support multiple datasets including road, road_waymo, and roadpp (both combine)
--TEST_DATASET  ---> Dataset use for evaluation in training MODE
--TRAIN_SUBSETS ---> It will be train in all cased except road where we have multiple splits
--SEQ_LEN       ---> We did experiments for sequence length of 8 but we support other lenths as well
--TEST_SEQ_LEN  ---> Test sequence length is for prediction of frames at a time we support mutliple lens and tested from 8 to 32.
--BATCH_SIZE    ---> The batch size depends upon the number of GPUs and/or your GPU memory, if your GPU memory is 24 GB we recommend a batch per GPU. For A100 80GB of GPU we tested upto 5 batchs per GPU.
```

- Training notes:
  * The VRAM required for a single batch is 16GB, in this case, you will need 4 GPUs (each with at least 16GB VRAM) to run training.
  * During training checkpoint is saved every epoch also log it's frame-level `frame-mean-ap` on a subset of validation split test.
  * Crucial parameters are `LR`, `MILESTONES`, `MAX_EPOCHS`, and `BATCH_SIZE` for training process.
  * `label_types` is very important variable, it defines label-types are being used for training and validation time it is bummed up by one with `ego-action` label type. It is created in `data\dataset.py` for each dataset separately and copied to `args` in `main.py`, further used at the time of evaluations.
  * Event detection and triplet detection is used interchangeably in this code base. 

## Testing and Building Tubes
To generate the tubes and evaluate them, first, you will need frame-level detection and link them. It is pretty simple in out case. Similar to training command, you can run following commands. These can run on single GPUs. 

There are various `MODEs` in `main.py`. You can do each step independently or together. At the moment `gen-dets` mode generates and evaluated frame-wise detection and finally performs tube building and evaluation.

For ROAD-Waymo dataset, run the following commands.

```
python main.py /home/user/ /home/user/  /home/user/kinetics-pt/ --MODE=gen_dets --MODEL_TYPE=I3D --DATASET=road_waymo --TEST_DATASET=road_waymo --VAL_SUBSETS=test --SEQ_LEN=8 --TEST_SEQ_LEN=8 --BATCH_SIZE=8 --LR=0.0041 
```

--TEST_DATASET specifies the dataset on which the model should be tested. Our Baseline support cross datasets train and where where the model is train on one dataset and tested on other. Our Baseline also support training and testing on ROAD and ROAD waymo (ROADPP) together.

- Testing notes
  * Evaluation can be done on single GPU for test sequence length up to 32  
  * Please go through the hypermeter in `main.py` to understand there functions.
  * After performing tubes a detection `.json` file is dumped, which is used for evaluation, see `tubes.py` for more detatils.
  * See `modules\evaluation.py` and `data\dataset.py` for frame-level and video-level evaluation code to compute `frame-mAP` and `video-mAP`.


## Performance

## TODO




##### Download pre-trained weights


## Citation
If this work has been helpful in your research please cite following articles:

    @ARTICLE {singh2022road,
    author = {Singh, Gurkirt and Akrigg, Stephen and Di Maio, Manuele and Fontana, Valentina and Alitappeh, Reza Javanmard and Saha, Suman and Jeddisaravi, Kossar and Yousefi, Farzad and Culley, Jacob and Nicholson, Tom and others},
    journal = {IEEE Transactions on Pattern Analysis & Machine Intelligence},
    title = {ROAD: The ROad event Awareness Dataset for autonomous Driving},
    year = {5555},
    volume = {},
    number = {01},
    issn = {1939-3539},
    pages = {1-1},
    keywords = {roads;autonomous vehicles;task analysis;videos;benchmark testing;decision making;vehicle dynamics},
    doi = {10.1109/TPAMI.2022.3150906},
    publisher = {IEEE Computer Society},
    address = {Los Alamitos, CA, USA},
    month = {feb}
    }



