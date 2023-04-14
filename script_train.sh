#!/bin/bash

agentness_th=0.5
LOGIC="Godel"

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py ../ ./experiments ../kinetics-pt/ --MODE=train --ARCH=resnet50 --MODEL_TYPE=I3D --DATASET=roadpp --TRAIN_SUBSETS=train --SEQ_LEN=8 --TEST_SEQ_LEN=8 --BATCH_SIZE=4 --LR=0.0041 --LOGIC=$LOGIC --req_loss_weight=10 --agentness_th=${agentness_th}

