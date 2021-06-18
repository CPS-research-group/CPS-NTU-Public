#!/usr/bin/python3
import numpy as np
import cv2
import os
from glob import glob

from sys import argv

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#font = {'weight' : 'normal', 'size':8}
#plt.rc('font', **font)
plt.rc('xtick', labelsize=1)
plt.rc('ytick', labelsize=1)

def makeplot(idx, datapoints, data_left, data_right, figsizeinpixel, title, threshold_h=0, threshold_v=0):

    # figsizeinpixel in (w+w,h). fig the same
    fig = plt.figure(figsize=(figsizeinpixel[0]/50, figsizeinpixel[1]/50), dpi=50)
    plt.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.02)

    ax_left_h = fig.add_subplot(221)
    ax_left_v = fig.add_subplot(223)
    ax_right_h = fig.add_subplot(222)
    ax_right_v = fig.add_subplot(224)

    ax_left_h.set_xlim([0, datapoints])
    ax_left_v.set_xlim([0, datapoints])
    ax_right_h.set_xlim([0, datapoints])
    ax_right_v.set_xlim([0, datapoints])

    scale = 6
    ax_left_h.set_ylim([0, threshold_h*scale])
    ax_left_v.set_ylim([0, threshold_v*scale])
    ax_right_h.set_ylim([0,  threshold_h*scale])
    ax_right_v.set_ylim([0,  threshold_v*scale])


    # plot if data available
    if len(data_left) > 0:
        # pad zero to future data point
        y_h = data_left["oodscore_h"][:idx]
        y_v = data_left["oodscore_v"][:idx]

        if "CPU" in title.upper():
            y_h = [y_h[i]*cpu_curve_factor for i in range(len(y_h))]
            y_v = [y_v[i]*cpu_curve_factor for i in range(len(y_v))]

        ax_left_h.plot(y_h, 'c')
        th_line = np.array([threshold_h for i in range(datapoints)])
        ax_left_h.plot(th_line,'r--', label="threshold-H")

        ax_left_v.plot(y_v, 'g')
        th_line = np.array([threshold_v for i in range(datapoints)])
        ax_left_v.plot(th_line,'r--', label="threshold-V")
        #ax_left_v.legend(loc="upper right")

    # plot if data available
    if len(data_right) > 0:
        # pad zero to future data point
        y_h = data_right["oodscore_h"][:idx]
        y_v = data_right["oodscore_v"][:idx]

        ax_right_h.plot(y_h, 'c', label="onbot-H")
        th_line = np.array([threshold_h for i in range(datapoints)])
        ax_right_h.plot(th_line,'r--', label="threshold-H")

        ax_right_v.plot(y_v, 'y', label="onbot-V")
        th_line = np.array([threshold_v for i in range(datapoints)])
        ax_right_v.plot(th_line,'r--', label="threshold-V")

    st = fig.suptitle(title)
    st.set_y(0.95)
    fig.subplots_adjust(top=0.85)

    # draw the renderer
    fig.canvas.draw( )
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring (fig.canvas.tostring_rgb(), dtype=np.uint8 )
    buf.shape = ( h, w, 3)  # (w,h)
    buf = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)

    # release memory
    plt.close(fig)
    return buf


def of_quiver(flow):

    (h,w,c) = flow.shape
    of_x, of_y = flow[...,0], flow[...,1]

    U = cv2.resize(of_x, None, fx=0.2, fy=0.2)
    V = cv2.resize(of_y, None, fx=0.2, fy=0.2)
    M = np.sqrt(np.add(U**2, V**2))

    X = np.arange(0, w, 5)
    Y = np.arange(0, h, 5)

    fig = plt.figure(figsize=(w/100, h/100), dpi=100)
    ax = fig.add_subplot(111)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)

    q = ax.quiver(X, Y, U, V, M, cmap=plt.cm.jet) #, scale=_quiver_scale)

    fig.canvas.draw( )
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring (fig.canvas.tostring_rgb(), dtype=np.uint8 )
    buf.shape = (h,w,3)
    #
    buf = buf[:]
    buf = np.flip(buf, axis=0)

    buf = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
    plt.close(fig) # release memory

    return buf

def parsecsv(csvfile):
    # csv format : timstamp,bool,float_H,float_v
    ret = {"timestamp":[], "ood":[], "oodscore_h":[], "oodscore_v":[]}
    with open(csvfile, 'r') as f:
        for l in f:
            tmp = l.split(",")
            # MAx timpstamp string length is 18
            ret["timestamp"].append(float(tmp[0]))
            ret["ood"].append(bool(tmp[1]))
            ret["oodscore_h"].append(float(tmp[2]))
            ret["oodscore_v"].append(float(tmp[3]))
    return ret

def viz_episode(episodeID, threshold_h, thresold_v, imagefile_ext="jpg"):
    #
    # episodeID format : episodes_dd.mm.yyyy/scene_desc/episode_suffix
    #

    ## Grab an ordered image sequence
    if imagesequence_padded:
        imagefiles = sorted(glob(episodeID + "/*.{}".format(imagefile_ext)))[1:]

    else:
        # To use when image file's timestamp was not padded to fixed length.
        imagefiles = []
        for imagefile in glob(episodeID + "/*.{}".format(imagefile_ext)):
            if len(imagefile.split("/")[-1].split(".")) == 2:
                imagefiles.append(imagefile)
            else:
                tmp = imagefile.split(".")
                # pad timestamp at left
                tmp[-2] = "{:010d}".format(int(tmp[-2]))
                newfile = ".".join(tmp)
                imagefiles.append(newfile)
                if newfile!=imagefile:
                    os.system("mv {} {}".format(imagefile, newfile))
        #
        imagefiles = sorted(imagefiles)[1:]


    flows = []
    rgbs = []
    im1 = None
    for i in range(len(imagefiles)):
	    im2 = cv2.imread(imagefiles[i])
        #print(im2.shape, imagefiles[i])

    if im2.shape[0] > 120 : im2 = cv2.resize(im2, None, (120,160))
        rgbs.append(im2)
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
        if im1 is not None:
            flow = cv2.calcOpticalFlowFarneback(im2, im1, None, pyr_scale = 0.5,
		                    levels = 1, iterations = 1,  				 # don't change these parameters
		                    winsize = 11, poly_n = 5, poly_sigma = 1.1,  # don't change these parameters
		                    flags = None)
            flows.append(flow)
        #
        im1 = im2


    ## fetch test results
    # Assume one csv file for each only
    results_offbot, results_onbot, results_cpu = [], [], []

    # Compare offbot and onbot resualts
    if len(results_cpu) == 0:
        for csvfile in glob(episodeID + "/*_onbot.csv"):
            results_onbot = parsecsv(csvfile)
            if len(results_onbot)==0:
                print("Miss ONbot result <- ".format(episodeID))
            elif len(results_onbot) != len(flows):
                print("[INFO]  ONbot result length doesn't match image sequence.")

    if len(results_cpu) > 0:
        datapoints = np.max([len(results_offbot["timestamp"]), len(results_cpu["timestamp"])])
    else:
        datapoints = np.max([len(results_offbot["timestamp"]), len(results_onbot["timestamp"])])


    ## make video
    videofile = episodeID + '.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # image in (h, w). sz in (w, h)
    (h,w,c) = rgbs[0].shape
    sz = (w+w, h+h)  #3-col in (rgb image, plots, of field)
    figsizeinpixel = (w+w,h)
    out = cv2.VideoWriter(videofile, fourcc, recordfps, sz, True)

    for i in range(len(flows)):
        if len(results_cpu) > 0:
            frame_plots = makeplot(i, datapoints, results_cpu, results_offbot, figsizeinpixel, \
                'Left : CPU  Right : offbot-TPU', threshold_h, thresold_v)
        else:
            frame_plots = makeplot(i, datapoints, results_offbot, results_onbot, figsizeinpixel, \
                'Left : offbot-TPU   Right : onbot-Realtime', threshold_h, thresold_v)

        frame_of = of_quiver(flows[i])
        frame_rgb = rgbs[i+1]

        frame_all = np.hstack((frame_rgb, frame_of))
        frame_all = np.vstack((frame_all, frame_plots))
        out.write(frame_all)

    out.release()
    cv2.destroyAllWindows()



cpu_curve_factor = 5*1e7
imagesequence_padded = False
recordfps = 5

if __name__ == "__main__":

    # Threshold values follows model that produces results. Set to None if don't know.
    threshold_h = None
    thresold_v = None

    episodeID = argv[1]
    viz_episode(episodeID, threshold_h, thresold_v)
