import argparse
import glob
from os.path import join
import cv2
import torch
import models.FlowNetS as FlowNetS
import torch.utils.data
import tensorflow as tf


# define program arguments

parser = argparse.ArgumentParser()
parser.add_argument('--video_directory', type=str, help="path to the directory containing the videos")
parser.add_argument('--total_epochs', default=10, type=int, help="number of epochs while training")
parser.add_argument('--batch_size', '-b', default=8, type=int, help="Batch size")
parser.add_argument('--save', '-s', default='./work', type=str, help='directory for saving')


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

        # TODO apply filter again?
        # frame = cv2.GaussianBlur(frame, (3, 3), 0)
        # frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
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


def convert_frame_to_tensor(frame):
    # Convert img to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return torch.tensor(rgb)


def flo_data_from_frames(frames, flo_data_path):
    global args
    """
    uses frame data to train a model

    :param frames: the array of frames used for training
    :param str flo_data_path: the directory where the flo data will be saved into
    """

    # load pre trained data
    if torch.cuda.is_available():
        pretrained = torch.load("./data/flownets_EPE1.951.pth.tar")
    else:
        pretrained = torch.load("./data/flownets_EPE1.951.pth.tar", map_location=torch.device('cpu'))

    # parse a model form pre trained data
    model = FlowNetS.flownets(pretrained)

    # generate data set to use for classification
    tensors = [convert_frame_to_tensor(x) for x in frames]
    data_set = []
    for i in range(len(frames) - 1):
        frame_pair = [tensors[i], tensors[i+1]]
        ground_truth = '' # TODO get from somewhere
        data_set.append([frame_pair, ground_truth])

    # generate stuff for every frame pair
    for i, (pair, target) in enumerate(data_set):
        cat = torch.cat(pair, 1).to()
        output = model(cat)

        print(output)


def train_strain_prediction(epoch, flo_data_path):
    """
    trains the model for predicting strains based on flo data

    :param int epoch: epoch size (how often this video will be parsed and classified)
    :param str flo_data_path: directory where the flo data was saved into
    """

    for i in range(epoch):
        # classify

        # compare results (by calculating loss)

        # save checkpoint
        pass


def main():
    print("getting all mp4 files in the video directory {} ...".format(args.video_directory))
    videos = glob.glob(join(args.video_directory, "*.mp4"))
    if len(videos) <= 0:
        print("No videos found. Please make sure the directory path is correct and the directory contains mp4 files.")
        return
    print("{} videos found.".format(len(videos)))

    # do the training for every video
    for i in range(len(videos)):
        print("processing video '{}' ...".format(videos[i]))

        # get frame data for the video - use frames wenji used in her code as well
        frames = processing_video(videos[i], 4500, 5000)

        # process these frames into flo data
        flo_path = join(args.save, "flo_data", videos[i])
        flo_data_from_frames(frames, flo_path)

        # train the model to predict strains
        train_strain_prediction(args.total_epochs, flo_path)

        print("processing '{}' finished.".format(videos[i]))


args = parser.parse_args()
main()
