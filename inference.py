import cv2
from PIL import Image


import os
import sys
import torch
import argparse
import numpy as np
from modules import utils
from train import train
from data.nuscene_loader import VideoDataset
from torchvision import transforms
import data.transforms as vtf
from models.retinanet import build_retinanet
from gen_dets import gen_dets, eval_framewise_dets
from tubes import build_eval_tubes
from val import val
import torch.utils.data as data_utils
from data import custum_collate

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def set_out_video(video_name):
    fps = 12
    video_width = 1600
    video_height = 900
    size = (video_width, video_height)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video = cv2.VideoWriter(video_name, fourcc, fps, size)
    return video


def main():
    parser = argparse.ArgumentParser(description='Training single stage FPN with OHEM, resnet as backbone')
    parser.add_argument('DATA_PATH', help='Location to root directory for dataset reading') # /mnt/mars-fast/datasets/
    parser.add_argument('SAVE_PATH', help='Location to root directory for saving checkpoint models') # /mnt/mars-alpha/
    parser.add_argument('MODEL_PATH',help='Location to root directory where kinetics pretrained models are stored')
    parser.add_argument('--MODE', default='gen_det',
                        help='MODE can be train, gen_dets, eval_frames, eval_tubes define SUBSETS accordingly, build tubes')
    # Name of backbone network, e.g. resnet18, resnet34, resnet50, resnet101 resnet152 are supported
    parser.add_argument('--ARCH', default='resnet50', 
                        type=str, help=' base arch')
    parser.add_argument('--MODEL_TYPE', default='I3D',
                        type=str, help=' base model')
    parser.add_argument('--ANCHOR_TYPE', default='RETINA',
                        type=str, help='type of anchors to be used in model')    
    parser.add_argument('--SEQ_LEN', default=8,
                        type=int, help='NUmber of input frames')
    parser.add_argument('--TEST_SEQ_LEN', default=8,
                        type=int, help='NUmber of input frames')
    parser.add_argument('--MIN_SEQ_STEP', default=1,
                        type=int, help='DIFFERENCE of gap between the frames of sequence')
    parser.add_argument('--MAX_SEQ_STEP', default=1,
                        type=int, help='DIFFERENCE of gap between the frames of sequence')
    # if output heads are have shared features or not: 0 is no-shareing else sharining enabled
    # parser.add_argument('--MULIT_SCALE', default=False, type=str2bool,help='perfrom multiscale training')
    parser.add_argument('--HEAD_LAYERS', default=3, 
                        type=int,help='0 mean no shareding more than 0 means shareing')
    parser.add_argument('--NUM_FEATURE_MAPS', default=5, 
                        type=int,help='0 mean no shareding more than 0 means shareing')
    parser.add_argument('--CLS_HEAD_TIME_SIZE', default=3, 
                        type=int, help='Temporal kernel size of classification head')
    parser.add_argument('--REG_HEAD_TIME_SIZE', default=3,
                    type=int, help='Temporal kernel size of regression head')
    
        #  Name of the dataset only voc or coco are supported
    parser.add_argument('--Domain_Adaptation', default=False, 
                        type=str2bool,help='Domain Adaptation')
    parser.add_argument('--DATASET', default='road', 
                        type=str,help='dataset being used')

    # Input size of image only 600 is supprted at the moment 
    parser.add_argument('--MIN_SIZE', default=600, 
                        type=int, help='Input Size for FPN')
    #  data loading argumnets
    parser.add_argument('-b','--BATCH_SIZE', default=4, 
                        type=int, help='Batch size for training')
    parser.add_argument('--TEST_BATCH_SIZE', default=1, 
                        type=int, help='Batch size for testing')
    # Number of worker to load data in parllel
    parser.add_argument('--NUM_WORKERS', '-j', default=8, 
                        type=int, help='Number of workers used in dataloading')
    # optimiser hyperparameters
    parser.add_argument('--OPTIM', default='SGD', 
                        type=str, help='Optimiser type')
    parser.add_argument('--RESUME', default=0, 
                        type=int, help='Resume from given epoch')
    parser.add_argument('--MAX_EPOCHS', default=30, 
                        type=int, help='Number of training epoc')
    parser.add_argument('-l','--LR', '--learning-rate', 
                        default=0.004225, type=float, help='initial learning rate')
    parser.add_argument('--MOMENTUM', default=0.9, 
                        type=float, help='momentum')
    parser.add_argument('--MILESTONES', default='20,25', 
                        type=str, help='Chnage the lr @')
    parser.add_argument('--GAMMA', default=0.1, 
                        type=float, help='Gamma update for SGD')
    parser.add_argument('--WEIGHT_DECAY', default=1e-4, 
                        type=float, help='Weight decay for SGD')
    
    # Freeze layers or not 
    parser.add_argument('--FBN','--FREEZE_BN', default=True, 
                        type=str2bool, help='freeze bn layers if true or else keep updating bn layers')
    parser.add_argument('--FREEZE_UPTO', default=1, 
                        type=int, help='layer group number in ResNet up to which needs to be frozen')
    
    # Loss function matching threshold
    parser.add_argument('--POSTIVE_THRESHOLD', default=0.5, 
                        type=float, help='Min threshold for Jaccard index for matching')
    parser.add_argument('--NEGTIVE_THRESHOLD', default=0.4,
                        type=float, help='Max threshold Jaccard index for matching')
    # Evaluation hyperparameters
    parser.add_argument('--EVAL_EPOCHS', default='30', 
                        type=str, help='eval epochs to test network on these epoch checkpoints usually the last epoch is used')
    parser.add_argument('--VAL_STEP', default=2, 
                        type=int, help='Number of training epoch before evaluation')
    parser.add_argument('--IOU_THRESH', default=0.5, 
                        type=float, help='Evaluation threshold for validation and for frame-wise mAP')
    parser.add_argument('--CONF_THRESH', default=0.5, 
                        type=float, help='Confidence threshold for to remove detection below given number')
    parser.add_argument('--NMS_THRESH', default=0.5, 
                        type=float, help='NMS threshold to apply nms at the time of validation')
    parser.add_argument('--TOPK', default=10, 
                        type=int, help='topk detection to keep for evaluation')
    parser.add_argument('--GEN_CONF_THRESH', default=0.5, 
                        type=float, help='Confidence threshold at the time of generation and dumping')
    parser.add_argument('--GEN_TOPK', default=100, 
                        type=int, help='topk at the time of generation')
    parser.add_argument('--GEN_NMS', default=0.5, 
                        type=float, help='NMS at the time of generation')
    parser.add_argument('--CLASSWISE_NMS', default=False, 
                        type=str2bool, help='apply classwise NMS/no tested properly')
    parser.add_argument('--JOINT_4M_MARGINALS', default=False, 
                        type=str2bool, help='generate score of joints i.e. duplexes or triplet by marginals like agents and actions scores')
    
    ## paths hyper parameters
    parser.add_argument('--COMPUTE_PATHS', default=False, 
                        type=str2bool, help=' COMPUTE_PATHS if set true then it overwrite existing ones')
    parser.add_argument('--PATHS_IOUTH', default=0.5,
                        type=float, help='Iou threshold for building paths to limit neighborhood search')
    parser.add_argument('--PATHS_COST_TYPE', default='score',
                        type=str, help='cost function type to use for matching, other options are scoreiou, iou')
    parser.add_argument('--PATHS_JUMP_GAP', default=4,
                        type=int, help='GAP allowed for a tube to be kept alive after no matching detection found')
    parser.add_argument('--PATHS_MIN_LEN', default=6,
                        type=int, help='minimum length of generated path')
    parser.add_argument('--PATHS_MINSCORE', default=0.1,
                        type=float, help='minimum score a path should have over its length')
    
    ## paths hyper parameters
    parser.add_argument('--COMPUTE_TUBES', default=False, type=str2bool, help='if set true then it overwrite existing tubes')
    parser.add_argument('--TUBES_ALPHA', default=0,
                        type=float, help='alpha cost for changeing the label')
    parser.add_argument('--TRIM_METHOD', default='none',
                        type=str, help='other one is indiv which works for UCF24')
    parser.add_argument('--TUBES_TOPK', default=10,
                        type=int, help='Number of labels to assign for a tube')
    parser.add_argument('--TUBES_MINLEN', default=5,
                        type=int, help='minimum length of a tube')
    parser.add_argument('--TUBES_EVAL_THRESHS', default='0.2,0.5',
                        type=str, help='evaluation threshold for checking tube overlap at evaluation time, one can provide as many as one wants')
    # parser.add_argument('--TRAIL_ID', default=0,
    #                     type=int, help='eval TUBES_Thtrshold at evaluation time')
    
    ###
    parser.add_argument('--LOG_START', default=10, 
                        type=int, help='start loging after k steps for text/tensorboard') 
    parser.add_argument('--LOG_STEP', default=10, 
                        type=int, help='Log every k steps for text/tensorboard')
    parser.add_argument('--TENSORBOARD', default=1,
                        type=str2bool, help='Use tensorboard for loss/evalaution visualization')

    # Program arguments
    parser.add_argument('--MAN_SEED', default=123, 
                        type=int, help='manualseed for reproduction')
    parser.add_argument('--MULTI_GPUS', default=True, type=str2bool, help='If  more than 0 then use all visible GPUs by default only one GPU used ') 

    # Use CUDA_VISIBLE_DEVICES=0,1,4,6 to select GPUs to use


    ## Parse arguments
    args = parser.parse_args()
    
    args.model_subtype = args.MODEL_TYPE.split('-')[0]
    args.MAX_SIZE = int(args.MIN_SIZE*1.40)

    vid_save_path = args.MODEL_PATH.split("/")[1]+"_"+os.path.split(args.MODEL_PATH)[-1][:-39]

    # args = utils.set_args(args) # set directories and SUBSETS fo datasets
    args.MULTI_GPUS = False if args.BATCH_SIZE == 1 else args.MULTI_GPUS
    ## set random seeds and global settings
    np.random.seed(args.MAN_SEED)
    torch.manual_seed(args.MAN_SEED)
    # torch.cuda.manual_seed_all(args.MAN_SEED)
    torch.set_default_tensor_type('torch.FloatTensor')

    # args = utils.create_exp_name(args)

    # utils.setup_logger(args)
    # logger = utils.get_logger(__name__)
    # logger.info(sys.version)


    args.SEQ_LEN = args.TEST_SEQ_LEN
    args.MAX_SEQ_STEP = 1
    # args.SUBSETS = args.TEST_SUBSETS
    full_test = True #args.MODE != 'train'
    args.skip_beggning = 0
    args.skip_ending = 0
    if args.MODEL_TYPE == 'I3D':
        args.skip_beggning = 2
        args.skip_ending = 2
    elif args.MODEL_TYPE != 'C2D':
        args.skip_beggning = 2

    skip_step = args.SEQ_LEN - args.skip_beggning


    val_transform = transforms.Compose([ 
                        vtf.ResizeClip_Fixed(args.MIN_SIZE, args.MAX_SIZE),
                        vtf.ToTensorStack(),
                        vtf.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])


    val_dataset = VideoDataset(args, transform=val_transform, full_test=full_test)
    print('Done Loading Dataset Validation Dataset')


    args.num_classes =  val_dataset.num_classes
    # one for objectness
    args.label_types = val_dataset.label_types
    args.num_label_types = val_dataset.num_label_types
    args.all_classes =  val_dataset.all_classes
    args.num_classes_list = val_dataset.num_classes_list
    args.num_ego_classes = val_dataset.num_ego_classes
    args.ego_classes = val_dataset.ego_classes
    args.head_size = 256





    net = build_retinanet(args).cuda()
    net = torch.nn.DataParallel(net)


    net.eval()
    args.MODEL_PATH = args.MODEL_PATH + '/model_000030.pth'
    print('Loaded model from :: '+args.MODEL_PATH)
    net.load_state_dict(torch.load(args.MODEL_PATH))
    


    val_data_loader = data_utils.DataLoader(val_dataset, 1, num_workers=args.NUM_WORKERS,
                                            shuffle=False, pin_memory=True)

    video = set_out_video(args.SAVE_PATH+'/'+vid_save_path+"_"+str(args.GEN_CONF_THRESH)+'_.MP4')
    activation = torch.nn.Sigmoid().cuda()
    with torch.no_grad():
        for val_itr, (images,img_names) in enumerate(val_data_loader):
            
            print(val_itr)
            images = images.cuda(0, non_blocking=True)
            decoded_boxes, confidence, ego_preds = net(images)
            confidence = activation(confidence)
            det_boxes = []
            for nlt in range(args.num_label_types):
                numc = args.num_classes_list[nlt]
                det_boxes.append([[] for _ in range(numc)])
            for s in range(args.SEQ_LEN):
                # print(img_names[s])
                image = cv2.imread(img_names[s][0])
                # image = cv2.resize(image,(width,height))
                org_height,org_width = image.shape[:2]
                # print(org_height,org_width)
                # print(wh)
                decoded_boxes_frame = decoded_boxes[0, s].clone()
                cc = 0

                decoded_boxes_batch = decoded_boxes[0,s]
                confidence_batch = confidence[0,s]
                scores = confidence_batch[:, 0].squeeze().clone()
                cls_dets, save_data = utils.filter_detections_for_dumping(args, scores, decoded_boxes_batch, confidence_batch)

                for ppred in save_data:
                    bbox = ppred[:4]
                    # print(bbox)
                    agent_lab_ind = max(ppred[5:15])
                    if agent_lab_ind > args.GEN_CONF_THRESH:
                        agent_lab = args.all_classes[1][np.argmax(ppred[5:15])]
                    else:
                        agent_lab = ''
                    
                    action_lab_ind = max(ppred[15:34])
                    if action_lab_ind > args.GEN_CONF_THRESH:
                        action_lab = args.all_classes[2][np.argmax(ppred[15:34])]
                    else:
                        action_lab = ''

                    loc_lab_ind = max(ppred[34:46])
                    if loc_lab_ind > args.GEN_CONF_THRESH:
                        loc_lab = args.all_classes[3][np.argmax(ppred[34:46])]
                    else:
                        loc_lab = ''

                    dup_lab_ind = max(ppred[46:85])
                    if dup_lab_ind > args.GEN_CONF_THRESH:
                        dup_lab = args.all_classes[4][np.argmax(ppred[46:85])]
                    else:
                        dup_lab = ''

                    trip_lab_ind = max(ppred[85:153])
                    if trip_lab_ind > args.GEN_CONF_THRESH:
                        trip_lab = args.all_classes[5][np.argmax(ppred[85:153])]
                    else:
                        trip_lab = ''
                    # print(agent_lab)
                    # print(action_lab)
                    # print(loc_lab)
                    # print(dup_lab)
                    # print(trip_lab)
                    bbox[0] = (bbox[0]/int(args.MAX_SIZE))*org_width # width x1
                    bbox[2] = (bbox[2]/int(args.MAX_SIZE))*org_width # width x2
                    bbox[1] = (bbox[1]/int(args.MIN_SIZE))*org_height # height y1
                    bbox[3] = (bbox[3]/int(args.MIN_SIZE))*org_height # height y2

                    cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
                    cv2.putText(image, agent_lab, (int(bbox[0]), int(bbox[3]+20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (11,12,255), 2)
                    cv2.putText(image, action_lab, (int(bbox[0]), int(bbox[3]+40)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (11,12,255), 2)
                    cv2.putText(image, loc_lab, (int(bbox[0]), int(bbox[3]+60)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (11,12,255), 2)
                    cv2.putText(image, dup_lab, (int(bbox[0]), int(bbox[3]+80)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (11,12,255), 2)
                    cv2.putText(image, trip_lab, (int(bbox[0]), int(bbox[3]+100)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (11,12,255), 2)

                video.write(image)
    video.release()
        
                        
if __name__ == "__main__":
    main()