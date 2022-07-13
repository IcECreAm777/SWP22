import argparse
import datetime
import glob
import ntpath
import os
import shutil
import time
from os.path import join, isfile
import cv2
import matplotlib.image
import numpy
import numpy as np
import torch
import torchvision
from torchvision.transforms import transforms
import models.FlowNetS as FlowNetS
import torch.utils.data
import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter

from Helper.listdataset import ListDataset
from Helper.multiscaleloss import multiscaleEPE, realEPE

import re
import flow_transforms
import flowiz as fz

# define program arguments

parser = argparse.ArgumentParser()
parser.add_argument('--video_directory', type=str, help="path to the directory containing the videos")
parser.add_argument('--total_epochs', default=10, type=int, help="number of epochs while training")
parser.add_argument('--batch_size', '-b', default=8, type=int, help="Batch size")
parser.add_argument('--save', '-s', default='./work', type=str, help='directory for saving')
parser.add_argument('--pretrained', '-p', type=str, help="path to a pretrained model")
parser.add_argument('--ground_truth', type=str, help="the path to the ground truth data")
parser.add_argument('--learning_rate', '-l', type=float, help="the learning rate used for the model", default=0.01)
parser.add_argument('--multiscale-weights', '-w', default=[0.005, 0.01, 0.02, 0.08, 0.32], type=float, nargs=5,
                    help='training weight for each scale, from highest resolution (flow2) to lowest (flow6)',
                    metavar=('W2', 'W3', 'W4', 'W5', 'W6'))
parser.add_argument('--sparse', action='store_true',
                    help='look for NaNs in target flow when computing EPE, avoid if flow is guaranteed to be dense')
parser.add_argument('--div-flow', default=20,
                    help='value by which flow will be divided. Original value is 20 but 1 with batchNorm gives good '
                         'results')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epoch-size', default=1000, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if set to 0)')
parser.add_argument('--max-values', default=False, type=bool, help="Whether to use only max values or all values")
parser.add_argument('--ground-truth-directory', default='./basedata/Normalized_XY', type=str,
                    help="The directory where the ground truth is located")

# global parameters

if __name__ == '__main__':
    timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
    save_path = os.path.join("./out/", timestamp)
    img_path = "./img/"

    number = 8

    position_x, position_y = 70 - number, 60 - number
    window_size_x, window_size_y = 64 + number * 2, 96 + number * 2
    color_map_x, color_map_y = 60, 160

    frame_start = 1600
    frame_end = 7200

    num_skipped_frames = 1

    train_writer = SummaryWriter(os.path.join(save_path, 'train'))
    test_writer = SummaryWriter(os.path.join(save_path, 'test'))
    output_writers = []
    for i in range(10):
        output_writers.append(SummaryWriter(os.path.join(save_path, 'test', str(i))))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print_freq = 10
    n_iter = int(5)


# Classes (got this class from https://github.com/ClementPinard/FlowNetPytorch)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.3f} ({:.3f})'.format(self.val, self.avg)


# functions

# (got this class from https://github.com/ClementPinard/FlowNetPytorch)
def flow2rgb(flow_map, max_value):
    flow_map_np = flow_map.detach().cpu().numpy()
    _, h, w = flow_map_np.shape
    flow_map_np[:, (flow_map_np[0] == 0) & (flow_map_np[1] == 0)] = float('nan')
    rgb_map = np.ones((3, h, w)).astype(np.float32)
    if max_value is not None:
        normalized_flow_map = flow_map_np / max_value
    else:
        normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    rgb_map[0] += normalized_flow_map[0]
    rgb_map[1] -= 0.5 * (normalized_flow_map[0] + normalized_flow_map[1])
    rgb_map[2] += normalized_flow_map[1]
    return rgb_map.clip(0, 1)


def processing_video(path, start, end):
    """
    takes a video and returns all the frames as images in the given time frame

    :param str path: path to the video file
    :param int start: start frame in the video
    :param int end: end frame of in the video
    :return: an array with the generated frames
    """

    frames = []
    cap = cv2.VideoCapture(path)
    num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == 0:
            break
        if num < start:
            num += 1
            continue
        # if num%2 == 1:
        #   num += 1
        #   continue
        if end != -1 and num >= end:
            break

        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)

        frame = frame[position_y: position_y + window_size_y, position_x: position_x + window_size_x]

        frames.append(frame)
        num += 1
    return frames


def save_checkpoint(data, save_path, filename):
    """
    saves the checkpoint data after an epoch.

    :param data: teh data to save
    :param str save_path: path of the directory the checkpoint will be saved in
    :param str filename: the filename for the checkpoint
    """

    path = join(save_path, filename)
    torch.save(data, path)
    print("checkpoint saved at {}".format(path))


def generate_training_data_set(video_name):
    """
    generates the training set for the data loader to use

    :param str video_name: the name of the current video
    """

    # read all the file path
    files = [f for f in os.listdir(img_path) if isfile(join(img_path, f))]

    # we assume the paths in the files array are always sorted

    # save file path to the images in the array
    data_set = []
    for i in range(len(files) - 1):
        # get the current and next frame number
        current_file_num = int(re.findall('[0-9]+', files[i])[0])
        next_file_num = int(re.findall('[0-9]+', files[i+1])[0])

        # only use frames that are directly linked (idk english)
        if next_file_num - current_file_num == 1:
            # generate the file names
            flo_file_name = "{}_{}.flo".format(video_name, current_file_num)
            ground_truth_path = join(args.ground_truth_directory, flo_file_name)

            # continue when no ground truth data was found
            if not os.path.exists(ground_truth_path):
                print("Ground truth file '{}' not found".format(ground_truth_path))
                continue

            # copy the ground truth file and get tha path to this file
            shutil.copy(ground_truth_path, img_path)

            # append file path to the data set
            frame_pair = [files[i], files[i + 1]]
            data_set.append([frame_pair, flo_file_name])

    return data_set


def open_flo_file(file):
    """
    opens the flo file at the given file path and prints out its values

    :param str file: the file path of the .flo file
    """
    with open(file, mode='r') as flo:
        tag = np.fromfile(flo, np.float32, count=1)[0]
        width = np.fromfile(flo, np.int32, count=1)[0]
        height = np.fromfile(flo, np.int32, count=1)[0]

        # print('tag', tag, 'width', width, 'height', height)

        nbands = 2
        tmp = np.fromfile(flo, np.float32, count=nbands * width * height)
        flow = np.resize(tmp, (int(height), int(width), int(nbands)))
        return flow


def generate_flo_file(index, data):
    """
    generates a .flo file for the given values

    :param int index: index for the comparison
    :param data: generated flo data by the model
    """

    # generate flo header properties
    tag = np.float32(202021.25)
    width = np.int32(window_size_x)
    height = np.int32(window_size_y)

    # write flo data into the file
    flo_path = '{}/{}.flo'.format(save_path, frame_start + index)
    with open(flo_path, 'wb') as flo:
        # write flo header
        flo.write(tag)
        flo.write(width)
        flo.write(height)

        # arrange data into flo format and write it into the file
        data.detach().numpy().astype(np.float32).tofile(flo)


def train_strain_prediction(epoch, data_set, video_name):
    global args
    """
    trains the model for predicting strains based on flo data

    :param int epoch: epoch size (how often this video will be parsed and classified)
    :param Any data_set: data set used to train the model
    :param str video_name: the name of the video which is currently processed
    """

    # init input transform
    input_transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0, 0, 0, 0], std=[255, 255, 255, 255]),
        transforms.Normalize(mean=[0.45, 0.432, 0.411, 0.4], std=[1, 1, 1, 1])
    ])

    # init target transform
    target_transform = transforms.Compose([
        flow_transforms.ArrayToTensor()
        # transforms.Normalize(mean=[0, 0], std=[args.div_flow, args.div_flow])
    ])

    # actually load stuff
    train_dataset = ListDataset(img_path, data_set, transform=input_transform, target_transform=target_transform)
    # TODO add test data set and use it in the val_loader

    # load images
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,                      # TODO use test data set here
        num_workers=args.workers, pin_memory=True, shuffle=False)

    # parse a model form pre-trained data (if any)
    data = None
    if args.pretrained:
        # load pre trained data
        if torch.cuda.is_available():
            data = torch.load(args.pretrained)
        else:
            data = torch.load(args.pretrained, map_location=torch.device('cpu'))

    # load model with data (if any)
    model = FlowNetS.flownets(data)

    # initialize optimizer and scheduler
    # param_groups = []
    # optimizer = torch.optim.Adam(param_groups, args.learning_rate)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.5)

    for i in range(epoch):
        # scheduler.step()

        # train model
        train_loss, train_epe = train(train_loader, model, None, i, train_writer)
        train_writer.add_scalar('mean EPE', train_epe, i)

        # test model
        pre_time = time.time()

        with torch.no_grad():
            if args.max_values:
                epe = validate_max_values(val_loader, model, i)
            else:
                epe = validate(val_loader, model, i)

        test_writer.add_scalar('mean EPE', epe, i)
        test_duration = time.time() - pre_time
        print("== test duration: {} == ".format(test_duration))

        # TODO classify as strain or not or save flo data to use in another program

        # save checkpoint
        save_checkpoint({
            'epoch': i + 1,
            'arch': 'flownets',     # TODO if other model are added, this is the place to save it
            'best_EPE': 0,          # TODO if best values are saved in the future, this is the place to save them
            'div_flow': args.div_flow
        }, save_path, "{}_checkpoint.pth.tar".format(video_name))


def train(train_loader, model, optimizer, epoch_index, writer):
    """
    actually trains the model with the given data

    :param train_loader: the data used for training
    :param model: the model to train
    :param optimizer:
    :param int epoch_index: the index of the current epoch
    :param writer: a summary writer to save results during training
    """

    global n_iter

    # initialize average meter
    losses = AverageMeter()
    flow2_epes = AverageMeter()

    # get the epoch size
    epoch_size = len(train_loader) if args.epoch_size == 0 else min(len(train_loader), args.epoch_size)

    # switch to train mode
    model.train()

    # generate stuff for every frame pair
    for i, (pair, target) in enumerate(train_loader):
        # let the model process the input
        cat = torch.cat(pair, 1).to(device)
        target = target.to(device)

        # compute output
        output = model(cat)

        # save first output as flo file for further calculation
        generate_flo_file(i, output[0])

        # calculate loss and epe
        loss = multiscaleEPE(output, target, weights=args.multiscale_weights, sparse=args.sparse)
        flow2_epe = args.div_flow * realEPE(output[0], target, sparse=args.sparse)

        # record loss and EPE
        losses.update(loss.item(), target.size(0))
        writer.add_scalar('train_loss', loss.item(), n_iter)
        flow2_epes.update(flow2_epe.item(), target.size(0))

        # print progress
        if i % print_freq == 0:
            print('Epoch: [{}][{}/{}]\t Loss {}\t EPE {}'.format(epoch_index, i, epoch_size, loss, flow2_epes))

        # increase loop variables
        n_iter += 1
        if i >= epoch_size:
            break

    return losses.avg, flow2_epes.avg


def validate(val_loader, model, epoch):

    flow2_epes = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (calculated, target) in enumerate(val_loader):
        target = target.to(device)
        calculated = torch.cat(calculated, 1).to(device)

        # compute output
        output = model(calculated)
        flow2_EPE = args.div_flow * realEPE(output, target, sparse=args.sparse)
        # record EPE
        flow2_epes.update(flow2_EPE.item(), target.size(0))

        if i < len(output_writers):  # log first output of first batches
            if epoch == args.start_epoch:
                mean_values = torch.tensor([0.45, 0.432, 0.411], dtype=calculated.dtype).view(3, 1, 1)
                output_writers[i].add_image('GroundTruth', flow2rgb(args.div_flow * target[0], max_value=10), 0)
                output_writers[i].add_image('Inputs', (calculated[0, :3].cpu() + mean_values).clamp(0, 1), 0)
                # output_writers[i].add_image('Inputs', (input[0, 3:].cpu() + mean_values).clamp(0, 1), 1) # TODO fix
            output_writers[i].add_image('FlowNet Outputs', flow2rgb(args.div_flow * output[0], max_value=10), epoch)

        if i % print_freq == 0:
            print('Test: [{}][{}/{}]\t EPE {}'
                  .format(epoch, i, len(val_loader), flow2_epes))

    print(' * EPE {:.3f}'.format(flow2_epes.avg))

    return flow2_epes.avg


def validate_max_values(val_loader, model, epoch):
    """ validates and tests the model outputs, but only uses the max values and saves scalars """

    flow2_epes = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (calculated, target) in enumerate(val_loader):
        target = target.to(device).max()
        calculated = torch.cat(calculated, 1).to(device)

        # compute output
        output = model(calculated).max()

        # loss
        flow2_epe = (target - output).abs()

        # record EPE
        flow2_epes.update(flow2_epe)

        if i < len(output_writers):  # log first output of first batches
            if epoch == args.start_epoch:
                output_writers[i].add_scalar('GroundTruth', target, 0)
                output_writers[i].add_scalar('Input', (calculated.cpu().max()).clamp(0, 1), 0)
            output_writers[i].add_scalar('FlowNet Outputs', output, epoch)

        if i % print_freq == 0:
            print('Test: [{}][{}/{}]\t EPE {}'
                  .format(epoch, i, len(val_loader), flow2_epes))

    print(' * EPE {:.3f}'.format(flow2_epes.avg))

    return flow2_epes.avg


def save_frames_as_png(frames):
    """
    takes the frames and saves them as pngs

    :param Any frames: array of frames to save
    """

    # create directory if needed
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    # save every frame
    if num_skipped_frames <= 2:
        for i in range(len(frames)):
            matplotlib.image.imsave("{}/{}.png".format(img_path, frame_start + i), frames[i])
        return

    # save only the frames needed and the successive
    for i in range(0, len(frames), num_skipped_frames):
        matplotlib.image.imsave("{}/{}.png".format(img_path, frame_start + i), frames[i])
        matplotlib.image.imsave("{}/{}.png".format(img_path, frame_start + i + 1), frames[i + 1])


def clear_work_dir():
    """clears the image directory to save system storage"""
    if os.path.exists(img_path):
        shutil.rmtree(img_path)

    os.mkdir(img_path)


def main():
    print("getting all mp4 files in the video directory {} ...".format(args.video_directory))
    videos = glob.glob(join(args.video_directory, "*.mp4"))
    if len(videos) <= 0:
        print("No videos found. Please make sure the directory path is correct and the directory contains mp4 files.")
        return
    print("{} videos found.".format(len(videos)))

    # do the training for every video
    for i, val in enumerate(videos):
        # get the video name only
        head, tail = ntpath.split(val)
        video_name = tail or ntpath.basename(head)

        print("processing video {}/{}: '{}' ...".format(i, len(videos), video_name))

        # clear work dir to make sure the correct data is loaded
        clear_work_dir()

        # get frame data for the video - use frames wenji used in her code as well
        frames = processing_video(val, frame_start, frame_end)
        save_frames_as_png(frames)

        # process these frames to make them usable in the training
        data = generate_training_data_set(video_name)

        # train the model to predict strains
        train_strain_prediction(args.total_epochs, data, video_name)

        print("processing '{}' finished.".format(videos[i]))

    clear_work_dir()


if __name__ == '__main__':
    args = parser.parse_args()
    main()
