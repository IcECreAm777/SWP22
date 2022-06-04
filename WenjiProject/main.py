import struct

import cv2
import os

import matplotlib.image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from numpy.core.multiarray import min_scalar_type
from scipy.ndimage.filters import gaussian_filter, median_filter
import math
from datetime import datetime
import flowiz as fz

# GLOBAL PARAMETERS

data_path = '/content/drive/MyDrive/Data/BA/Image_Data/'
save_path = '/content/drive/MyDrive/Data/BA/'

videos = ['/316_7_1_1.mp4']

number = 8

position_x, position_y = 70 - number, 60 - number
window_size_x, window_size_y = 64 + number * 2, 96 + number * 2
color_map_x, color_map_y = 60, 160

grid = 1
sigma = 40
kernel_size = 11

save_field = False
save_dis_video = False
save_strain_video = True

# lk_params = dict(winSize=(63, 63),
#         maxLevel=10,
#         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.03))
lk_params = dict(winSize=(31, 31),
                 maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))

########################################################################################################################
# own code additions
########################################################################################################################

now = datetime.now()

num_skipped_frames = 10

frame_start = 1600
frame_end = 7200

ignore_horizontal_movement = True

# generate paths
dir_path = "./out/{}_{}_{}".format(now.hour, now.minute, now.second)
txt_path = '{}/processed.txt'.format(dir_path)

# create directory if it doesn't exist
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

# create directory if it doesn't exist
if not os.path.exists(txt_path):
    open(txt_path, 'x')


def process_displacement_data(index, displacement_x, displacement_y):
    """
    Processes the calculated displacement data into ground truth for the flow net project

    :param int index: frame index
    :param displacement_x: all displacement data in the x direction
    :param displacement_y: all displacement data in the y direction
    """

    # generate the output string and print it to console
    output_string = "frame: {} | displacement_x = {} {} | displacement_y = {} {}".format(
        frame_start + index, displacement_x.min(), displacement_x.max(), displacement_y.min(), displacement_y.max())
    print(output_string)

    # save data into file
    with open(txt_path, 'a') as f:
        f.write(output_string + "\n")

    generate_flo_file(index, displacement_x, displacement_y)


def open_flo_file(file):
    """
    opens the flo file at the given file path and prints out its values

    :param str file: the file path of the .flo file
    """
    with open(file, mode='r') as flo:
        tag = np.fromfile(flo, np.float32, count=1)[0]
        width = np.fromfile(flo, np.int32, count=1)[0]
        height = np.fromfile(flo, np.int32, count=1)[0]

        print('tag', tag, 'width', width, 'height', height)

        nbands = 2
        tmp = np.fromfile(flo, np.float32, count=nbands * width * height)
        flow = np.resize(tmp, (int(height), int(width), int(nbands)))
        print(flow)


def generate_flo_file(index, displacement_x, displacement_y):
    """
    generates a .flo file for the given values

    :param int index: index for the comparison
    :param displacement_x: all displacement data in the x direction
    :param displacement_y: all displacement data in the y direction
    """

    # generate flo header properties
    tag = np.float32(202021.25)
    width = np.int32(window_size_x)
    height = np.int32(window_size_y)

    # write flo data into the file
    flo_path = '{}/{}.flo'.format(dir_path, frame_start + index)
    with open(flo_path, 'wb') as flo:
        # write flo header
        flo.write(tag)
        flo.write(width)
        flo.write(height)

        # arrange data into flo format and write it into the file
        horizontal = displacement_x if not ignore_horizontal_movement else 0
        tmp = np.zeros((height, width * 2))
        tmp[:, np.arange(width) * 2] = horizontal
        tmp[:, np.arange(width) * 2 + 1] = displacement_y
        tmp.astype(np.float32).tofile(flo)

    # save flow as png
    img = fz.convert_from_file(flo_path)
    matplotlib.image.imsave("{}/{}.png".format(dir_path, frame_start + index), img)


# open_flo_file('{}/{}.flo'.format(dir_path, index))

########################################################################################################################
# This is the google collab file wenji created but as a single python file to run on a local machine
########################################################################################################################


def processing_video(path, start, end):
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
        frames.append(frame)
        num += 1
    return frames


def calculating_dist2(frames):
    p0 = np.array([[[x, y]] for y in range(0, window_size_y, grid)
                   for x in range(0, window_size_x, grid)], dtype=np.float32)
    size = p0.shape

    minx, maxx, miny, maxy = 0, 0, 0, 0
    displacements_x = np.zeros((len(frames) - 1, window_size_y, window_size_x))
    displacements_y = np.zeros((len(frames) - 1, window_size_y, window_size_x))

    for index in range(len(frames) - 1):
        pre_frame = frames[index]
        cur_frame = frames[index + 1]
        pre_cut = pre_frame[position_y:position_y + window_size_y, position_x:position_x + window_size_x]
        cur_cut = cur_frame[position_y:position_y + window_size_y, position_x:position_x + window_size_x]
        pre_gray = cv2.cvtColor(pre_cut, cv2.COLOR_BGR2GRAY)
        cur_gray = cv2.cvtColor(cur_cut, cv2.COLOR_BGR2GRAY)

        invalid_col = set()
        invalid_row = set()
        p1, st, err = cv2.calcOpticalFlowPyrLK(pre_gray, cur_gray, p0, None, **lk_params)
        p1.shape = (int(window_size_y / grid), int(window_size_x / grid), 2)
        p0.shape = (int(window_size_y / grid), int(window_size_x / grid), 2)

        for i in range(p1.shape[0]):
            for j in range(p1.shape[1]):
                x = int(p1[i][j][0])
                y = int(p1[i][j][1])
                if x < cur_gray.shape[1] and y < cur_gray.shape[0] and x >= 0 and y >= 0:
                    displacements_x[index][y][x] = p1[i][j][0] - p0[i][j][0]  # np.abs was here before
                    displacements_y[index][y][x] = p1[i][j][1] - p0[i][j][1]  # np.abs was here before
                # replace the invalid point
                if x >= cur_gray.shape[1]:
                    invalid_col.add(j)
                if y >= cur_gray.shape[0] or y < 0:
                    invalid_row.add(i)

        invalid_col = list(invalid_col)
        invalid_row = list(invalid_row)

        if bool(invalid_col):
            new_points = np.array([[[x, y]] for y in range(0, window_size_y, grid)
                                   for x in range(0, len(invalid_col) * grid, grid)], dtype=np.float32)
            # print(invalid_col,new_points.shape)
            new_points.shape = (int(window_size_y / grid), len(invalid_col), 2)
            p1 = np.delete(p1, invalid_col, axis=1)
            p1 = np.column_stack((new_points, p1))

        if bool(invalid_row):
            new_points = np.array([[[x, y]] for y in range(0, len(invalid_row) * grid, grid)
                                   for x in range(0, window_size_x, grid)], dtype=np.float32)
            # print(invalid_row,new_points.shape)
            new_points.shape = (len(invalid_row), int(window_size_x / grid), 2)
            p1 = np.delete(p1, invalid_row, axis=0)
            p1 = np.row_stack((new_points, p1))

        # displacements_x[index] = gaussian_filter(displacements_x[index],sigma=11,mode='constant',cval=0)
        # displacements_y[index] = gaussian_filter(displacements_y[index],sigma=11,mode='constant',cval=0)
        # dis_x = cv2.medianBlur(displacements_x[index], 3)
        # dis_y = cv2.medianBlur(displacements_y[index], 3)

        # print("frams: ", index, " invalid: ", invalid_col)
        min1 = displacements_x[index].min()
        max1 = displacements_x[index].max()
        min2 = displacements_y[index].min()
        max2 = displacements_y[index].max()
        if min1 < minx:
            minx = min1
        if max1 > maxx:
            maxx = max1
        if min2 < miny:
            miny = min2
        if max2 > maxy:
            maxy = max2
        # print('index:', index, "displacement_x = ", min1, max1, "displacement_y", min2, max2)
        # print("minx = ",minx,"maxx = ",maxx,"miny = ",miny,"maxy = ",maxy)

        if index % num_skipped_frames == 0:
            process_displacement_data(index, displacements_x[index], displacements_y[index])

        p0 = p1.reshape(size[0], size[1], size[2])

    return displacements_x, displacements_y


def calculating_strain(vd, displacements_x, displacements_y):
    Exx_res = np.zeros((displacements_x.shape[0], displacements_x.shape[1], displacements_x.shape[2]))
    Exy_res = np.zeros((displacements_x.shape[0], displacements_x.shape[1], displacements_x.shape[2]))
    Eyy_res = np.zeros((displacements_y.shape[0], displacements_y.shape[1], displacements_y.shape[2]))
    Eeqv_res = np.zeros((displacements_y.shape[0], displacements_y.shape[1], displacements_y.shape[2]))

    for index in range(displacements_x.shape[0]):

        gradientx = np.gradient(displacements_x[index])
        ux_src = gradientx[1]
        uy_src = gradientx[0]

        gradienty = np.gradient(displacements_y[index])
        vx_src = gradienty[1]
        vy_src = gradienty[0]

        kernel_size = 3
        ux = cv2.GaussianBlur(ux_src, (kernel_size, kernel_size), 0)
        uy = cv2.GaussianBlur(uy_src, (kernel_size, kernel_size), 0)
        vx = cv2.GaussianBlur(vx_src, (kernel_size, kernel_size), 0)
        vy = cv2.GaussianBlur(vy_src, (kernel_size, kernel_size), 0)

        Exx = (ux + 0.5 * (ux ** 2 + vx ** 2)) * 1.0
        Exy = 0.5 * (uy + vx + ux * uy + vx * vy) * 1.0
        Eyy = (vy + 0.5 * (vx ** 2 + vy ** 2)) * 1.0
        Eeqv = np.sqrt(Exx ** 2 + Eyy ** 2 - Exx * Eyy + 3 * Exy ** 2)

        # Exx = np.zeros((displacements_x.shape[1],displacements_x.shape[2]))
        # Exy = np.zeros((displacements_x.shape[1],displacements_x.shape[2]))
        # Eyy = np.zeros((displacements_x.shape[1],displacements_x.shape[2]))
        # Eeqv = np.zeros((displacements_x.shape[1],displacements_x.shape[2]))
        # for i in range(displacements_x.shape[1]):
        #   for j in range(displacements_x.shape[2]):
        #     Exx[i][j] = ux[i][j]+0.5*(pow(ux[i][j],2)+pow(vx[i][j],2))
        #     Exy[i][j] = 0.5*(uy[i][j]+vx[i][j]+ux[i][j]*uy[i][j]+vx[i][j]*vy[i][j])
        #     Eyy[i][j] = vy[i][j]+0.5*(pow(vx[i][j],2)+pow(vy[i][j],2))
        #     Eeqv[i][j] = math.sqrt(pow(Exx[i][j],2)+pow(Eyy[i][j],2)-Exx[i][j]*Eyy[i][j]+3*pow(Exy[i][j],2))

        Exx_res[index] = cv2.GaussianBlur(Exx, (kernel_size, kernel_size), 0)
        Exy_res[index] = cv2.GaussianBlur(Exy, (kernel_size, kernel_size), 0)
        Eyy_res[index] = cv2.GaussianBlur(Eyy, (kernel_size, kernel_size), 0)
        Eeqv_res[index] = cv2.GaussianBlur(Eeqv, (kernel_size, kernel_size), 0)

        print('index ', index, 'max = ', np.max(Eeqv_res))

        if save_field:
            Eeqv_res_path = os.path.join(save_path, 'Strain', vd)
            if not os.path.exists(Eeqv_res_path):
                os.makedirs(Eeqv_res_path)
            np.savetxt(os.path.join(Eeqv_res_path, '{:04d}'.format(index + 1)) + '.csv', Eeqv,
                       delimiter=',', fmt='%.2f')

    return (Exx_res, Exy_res, Eyy_res, Eeqv_res)


def video_paramter(path, size):
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    videoWriter = cv2.VideoWriter(path, fourcc, fps, size)
    return videoWriter


def Color(v, vmin, vmax):
    r, g, b = 1, 1, 1
    dv = vmax - vmin
    if v > vmax:
        v = vmax

    if v < (vmin + 0.25 * dv):
        r = 0
        g = 4 * (v - vmin) / dv
    elif v < (vmin + 0.5 * dv):
        r = 0
        b = 1 + 4 * (vmin + 0.25 * dv - v) / dv
    elif v < (vmin + 0.75 * dv):
        r = 4 * (v - vmin - 0.5 * dv) / dv
        b = 0
    else:
        g = 1 + 4 * (vmin + 0.75 * dv - v) / dv
        b = 0

    return int(b * 255), int(g * 255), int(r * 255)


def getColorMap(min, max):
    colorMap = np.ones((color_map_y, color_map_x, 3))
    colorMap = colorMap * 255
    y = 10
    # step = (max-min)/220
    step = (max - min) / 110
    step = int(step * 1000.0) / 1000.0
    for i in np.arange(max, min, -step):
        color = Color(i, 0, max - min)
        cv2.line(colorMap, (0, y), (20, y), color, 1)
        y += 1
    cv2.putText(colorMap, '[mm]', (0, 8), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
    height = y

    for i in np.arange(min, max, (max - min) / 10):
        scale = int(i * 100.0) / 100.0
        cv2.line(colorMap, (20, y), (25, y), (0, 0, 0), 1)
        cv2.putText(colorMap, str(scale), (26, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
        y -= int(height / 10)

    # cv2.line(colorMap,(20, 10+y),(25, 10+y),(0,0,0), 1)
    # cv2.putText(colorMap,"0",(26,13+y),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,0,0),1)

    return colorMap


def generating_video1(frames, displacements_x, displacements_y, videoWriter):
    min_x = displacements_x.min()
    max_x = displacements_x.max()
    min_y = displacements_y.min()
    max_y = displacements_y.max()
    print(vd, ' min x displacement is ', min_x, ' max is ', max_x)
    print(vd, ' min y displacement is ', min_y, ' max is ', max_y)

    colorMapx = getColorMap(min_x, max_x)
    colorMapy = getColorMap(min_y, max_y)

    for index in range(len(frames) - 1):
        frame = frames[index + 1]
        frame_x = frame.copy()
        frame_y = frame.copy()

        frame_x[10:color_map_y + 10, frames[0].shape[1] - color_map_x:] = colorMapx
        frame_y[10:color_map_y + 10, frames[0].shape[1] - color_map_x:] = colorMapy

        frame_cut_x = frame_x[position_y:position_y + window_size_y, position_x:position_x + window_size_x]
        frame_cut_y = frame_y[position_y:position_y + window_size_y, position_x:position_x + window_size_x]

        false_color_x = np.zeros_like(frame_cut_x)
        false_color_y = np.zeros_like(frame_cut_y)

        for i in range(displacements_x[index].shape[1]):
            for j in range(displacements_x[index].shape[0]):
                false_color_x[j][i] = Color(displacements_x[index][j][i], 0, max_x - min_x)
                false_color_y[j][i] = Color(displacements_y[index][j][i], 0, max_y - min_y)

        imgx = cv2.addWeighted(frame_cut_x, 0.2, false_color_x, 0.8, 0)
        imgy = cv2.addWeighted(frame_cut_y, 0.2, false_color_y, 0.8, 0)
        frame_x[position_y:position_y + window_size_y, position_x:position_x + window_size_x] = imgx
        frame_y[position_y:position_y + window_size_y, position_x:position_x + window_size_x] = imgy

        frame_dis = np.concatenate((frame_x, frame_y), axis=0)
        videoWriter.write(frame_dis)

        # plt.imshow(frame_dis)
        # break
        # im = Image.fromarray(np.uint8(frame_c))
        # im.save("/content/colorMap.jpg")


def generating_video2(frames, Eeqv_res, videoWriter):
    del_list = list(range(number))
    Eeqv_res_1 = np.delete(Eeqv_res, del_list, axis=1)
    Eeqv_res_1 = np.delete(Eeqv_res_1, del_list, axis=2)

    del_list2 = list(range(Eeqv_res_1.shape[1] - number, Eeqv_res_1.shape[1], 1))
    Eeqv_res_1 = np.delete(Eeqv_res_1, del_list2, axis=1)

    del_list3 = list(range(Eeqv_res_1.shape[2] - number, Eeqv_res_1.shape[2], 1))
    Eeqv_res_1 = np.delete(Eeqv_res_1, del_list3, axis=2)

    dv = Eeqv_res_1.max() - Eeqv_res_1.min()
    eqv_max = Eeqv_res_1.max() - dv * 0.1
    eqv_min = Eeqv_res_1.min() + dv * 0.3

    colorMapeqv = getColorMap(Eeqv_res_1.min(), Eeqv_res_1.max())
    for index in range(len(frames) - 1):
        frame = frames[index + 1]
        frame_eqv = frame.copy()
        # frame_eqv[:color_map_y,frames[0].shape[1]-color_map_x:] = colorMapeqv
        frame_eqv[position_y + 2 * number:color_map_y + 2 * number + position_y,
        position_x + window_size_x + 10:position_x + window_size_x + 10 + color_map_x] = colorMapeqv
        # frame_cut_eqv = frame_eqv[position_y:position_y+window_size_y,position_x:position_x+window_size_x]
        frame_cut_eqv = frame_eqv[position_y + number * 2:position_y + window_size_y,
                        position_x + number * 2:position_x + window_size_x]
        false_color_eqv = np.zeros_like(frame_cut_eqv)

        for i in range(Eeqv_res_1[index].shape[1]):
            for j in range(Eeqv_res_1[index].shape[0]):
                data = Eeqv_res_1[index][j][i]
                if (Eeqv_res_1[index][j][i] > eqv_max):
                    data = eqv_max
                if (Eeqv_res_1[index][j][i] < eqv_min):
                    data = 0
                false_color_eqv[j][i] = Color(data, 0, eqv_max - eqv_min)

        # imgeqv = cv2.addWeighted(frame_cut_eqv,0.3,false_color_eqv,0.7,0)
        # frame_eqv[position_y:position_y+window_size_y,position_x:position_x+window_size_x] = imgeqv
        # frame_eqv[position_y+number*2:position_y+window_size_y,position_x+number*2:position_x+window_size_x] = imgeqv
        # videoWriter.write(frame_eqv)

        videoWriter.write(false_color_eqv)

        for vd in videos:
            # load video
            vd_path = os.path.join(data_path, vd + '.mp4')
            print(vd_path)
            frames = processing_video(vd_path, 460, 1660)
            # frames = processing_video('/content/drive/MyDrive/Data/BA/VD/316_7_2_1.mp4',1500,3500)
            print(vd, ' includes ', len(frames), ' frames')

            # # calculate displacement
            displacements_x, displacements_y = calculating_dist2(frames)

            if save_dis_video:
                size = (frames[0].shape[1], frames[0].shape[0] * 2)
                save_dis_path = os.path.join(save_path, 'disVideo', vd + '_dis.avi')
                disVideoWriter = video_paramter(save_dis_path, size)
                generating_video1(frames, displacements_x, displacements_y, disVideoWriter)
                disVideoWriter.release()

            # calculate strain
            Exx_res, Exy_res, Eyy_res, Eeqv_res = calculating_strain(vd, displacements_x, displacements_y)

            if save_strain_video:
                # size = (frames[0].shape[1], frames[0].shape[0]*4)
                size = (frames[0].shape[1], frames[0].shape[0])
                save_strain_path = os.path.join(save_path, 'strainVideo', vd + '_strain.avi')
                strainVideoWriter = video_paramter(save_strain_path, (128, 128))
                generating_video2(frames, Eeqv_res, strainVideoWriter)
                strainVideoWriter.release()


# MAIN

frames = processing_video('E:/Files/Uni/SWP/videos/316_7_1_1.mp4', frame_start, frame_end)

# for i in range(len(frames))

calculating_dist2(frames)

image = frames[10].copy()
cv2.rectangle(image, (70, 60), (80 + 128, 60 + 96), (255, 0, 0), 1)
plt.imshow(image)
