#!/bin/bash

CCN_CONSTRAINTS="./constraints/full"
CCN_CENTRALITY="rev-custom2"
CCN_CUSTOM_ORDER="41,42,43,44,45,46,47,40,38,23,39,27,11,33,28,5,22,21,24,20,31,16,26,15,9,6,18,34,19,30,25,3,12,2,10,36,8,35,4,32,1,37,29,14,13,17,0"
CCN_NUM_CLASSES=48
CLIP=10.0

CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py ../ ./experiments/ ../kinetics-pt/ --MODE=train --ARCH=resnet50 --MODEL_TYPE=I3D --DATASET=roadpp --TRAIN_SUBSETS=train --SEQ_LEN=8 --TEST_SEQ_LEN=8 --BATCH_SIZE=4 --LR=0.0041 --CCN_CONSTRAINTS=${CCN_CONSTRAINTS} --CCN_CENTRALITY=${CCN_CENTRALITY} --CCN_CUSTOM_ORDER=${CCN_CUSTOM_ORDER} --CCN_NUM_CLASSES=${CCN_NUM_CLASSES} --CLIP=${CLIP}
