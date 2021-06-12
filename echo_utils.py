import os
import csv
import cv2
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import transforms
import PIL
from PIL import ImageOps
from keras.preprocessing.image import load_img, img_to_array, array_to_img
import time


echonet_dir = Path('../Echonet/')
video_dir = echonet_dir / 'Videos/'
label_trace_path = echonet_dir / 'Labels' / 'VolumeTracings.csv'
file_list_path = echonet_dir / 'Labels' / 'FileList.csv'
train_path = video_dir / 'train/'
test_path = video_dir / 'test/'
val_path = video_dir / 'val/'


def load_video(path, filename):
    """
    Function to load a video from a directory,
    exctract the stream of frames and transpose into numpy array
    with shape (3, FRAMECOUNT, 112, 112)
    :param path: directory containing videos
    :param filename: name of video
    :return: numpy array containing frames
    """
    vpath = (path + "/" + filename)
    print("vpath", vpath)
    if not os.path.exists(vpath):
        raise FileNotFoundError
    cap = cv2.VideoCapture(vpath)

    f_c = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vframes = np.zeros((f_c, f_w, f_h, 3), np.uint8)

    for i in range(f_c):
        ret, frame = cap.read()
        if not ret:
            raise ValueError("")
        vframes[i] = frame

    vframes = vframes.transpose((3, 0, 1, 2))

    return vframes, filename


def load_filelist_data(path):
    """
        Function to extract Filelist data and put into numpy array
        :param path: directory containing FileList.csv file
        :return: numpy array containing fileslist data
    """
    with open(path, newline='') as csvfile:
        print("filelist data loading...")
        header = csvfile.readline().strip().split(',')
        assert header == ["FileName", "EF", "ESV", "EDV",
                          "FrameHeight", "FrameWidth", "FPS",
                          "NumberOfFrames", "Split"]
        print("filelist data loaded.")
        filereader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        filelist = []
        for row in filereader:
            split_row = row[0].split(',')
            assert (len(split_row) == 9)
            filelist.append(split_row)
        filelist = np.array(filelist)
    csvfile.close()
    return filelist


def load_tracing_data(path, fl):
    """
        Function to extract VolumeTracing data and put into numpy array
        :param path: directory containing VolumeTracings.csv file
        :return: numpy array containing fileslist data
    """
    raw_tracings = []
    tracings = []
    for f in fl[0:-6]:
        vname = [f[0]]
        tracings.append(vname)
    # check filelist [] and tracings [] are the same length
    assert len(tracings) == len(fl[0:-6])

    with open(path, newline='') as csvfile:
        header = csvfile.readline().strip().split(',')
        assert header == ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]
        filereader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in filereader:
            srow = row[0].split(',')
            # check length of each row is same as the number of...
            # columns defined in header
            assert len(srow) == len(header)
            raw_tracings.append(srow)
    csvfile.close()

    i = 0
    for trace in tracings:
        print(i)
        for raw in raw_tracings[i:]:
            if raw[0] == trace[0]:
                coords = raw[1:6]
                trace.append(coords)
                i += 1

    return tracings


# Function to get EDV and ESV video frame index...
# from VolumeTracings.csv
def get_cadio_idx_csv(vname, path):
    a, b = "", ""
    with open(path, newline='') as csvfile:
        header = csvfile.readline().strip().split('.')
        assert header == ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]
        filereader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for irow in filereader:
            srow = irow[0].split(',')
            if srow[0] == vname:
                srow[-1] = a
        for jrow in filereader:
            s_row = jrow[0].split(',')
            if jrow[0] == vname and jrow[-1] != a:
                s_row[-1] = b
        edv = ""
        esv = ""
        if a < b:
            edv = a
            esv = b
        elif b < a:
            edv = b
            esv = a
    csvfile.close()


# Function to return EDV and ESV frames from...
# all video frames.
def get_cardiac_frames(frames, filename, tracings):
    assert len(tracings) == 10024
    v_tracings = []
    for tracing in tracings:
        if tracing[0] == filename:
            v_tracings.append(tracing)
            break
        else:
            pass
    [v_tracings] = v_tracings

    d_idx = int(v_tracings[1][-1])
    s_idx = int(v_tracings[-1][-1])
    edv = frames[0][d_idx]
    esv = frames[0][s_idx]

    return edv, esv


# Function to take a frame and save as image.png
def gen_original(frames, filename, tracings, rotation=0):
    original = []
    # f_count = len(frames[0])  # frame count of video
    f_len = len(frames[0][0])
    name = filename.split('.')
    # assertion based on the 6 last rows/videos...
    # present in filelist.csv and the lack of...
    # tracings coordinates for said videos
    assert len(tracings) == 10024
    base = np.zeros((112, 112), dtype=np.uint8)
    edv, esv = get_cardiac_frames(frames, filename, tracings)
    # Create polygon mask and image
    dmask_fig, ax = plt.subplots(figsize=(f_len, f_len),
                                 dpi=1,
                                 frameon=False,
                                 tight_layout=True)
    if rotation != 0:
        tr = transforms.Affine2D().rotate_deg(rotation)
        plt.imshow(base, interpolation='none', cmap='gray')
        plt.imshow(edv, interpolation='none', cmap='gray', transform=tr + ax.transData)
    else:
        plt.imshow(edv, interpolation='none', cmap='gray')
    ax.axis('off')
    # plot data as image
    dmask_fig.savefig('esv_original_rot3_' + name[0] + '.png')
    plt.close(dmask_fig)
    return original


# Function to generate mask from BDS tracings
def gen_mask(frames, filename, tracings, vframe='edv', overlay=False):
    """
    Function to create masks and save as .png file.
    Masks created using a patched plt.Polygon, this must be a closed polygon.
    Order of tracings points starts from the top of the LV length,
    this being the first row in each videos EDV and ESV tracings within VolumeTracings.csv.
    The points then proceed counter-clockwise around the BDS tracing and end with
    the last point being the same as the first, the top of the LV length,
    forming a closed polygon.
    :param frames: list, of video frames
    :param filename: string, of video filename
    :param tracings: list, of video volume tracings
    :param vframe: string, to specify End Dystolic or End Systolic volume. Default EDV
    :param overlay: Bool, to overlay the mask with the original frame. Default False
    :return: nothing: saves plt.fig to home dir
    """

    # f_count = len(frames[0])  # frame count of video
    f_len = len(frames[0][0])
    name = filename.split('.')
    # assertion based on the 6 last rows/videos...
    # present in filelist.csv and the lack of...
    # tracings coordinates for said videos
    assert len(tracings) == 10024
    v_tracings = []
    for tracing in tracings:
        if tracing[0] == filename:
            v_tracings.append(tracing)
            break
        else:
            pass
    [v_tracings] = v_tracings

    edv, esv = get_cardiac_frames(frames, filename, tracings)
    _overlay = edv
    d_idx = int(v_tracings[1][-1])
    s_idx = int(v_tracings[-1][-1])

    d_coords = []
    s_coords = []
    for trace in v_tracings[1:]:
        if int(trace[-1]) == d_idx:
            d_coords.append(trace)
        elif int(trace[-1]) == s_idx:
            s_coords.append(trace)
    # check LV length coords are the right way around...
    # LV length is first row in tracings
    if float(d_coords[0][1]) > float(d_coords[0][3]):
        d_coords[0][0], d_coords[0][2] = d_coords[0][2], d_coords[0][0]
        d_coords[0][1], d_coords[0][3] = d_coords[0][3], d_coords[0][1]
    if float(s_coords[0][1]) > float(s_coords[0][3]):
        s_coords[0][0], s_coords[0][2] = s_coords[0][2], s_coords[0][0]
        s_coords[0][1], s_coords[0][3] = s_coords[0][3], s_coords[0][1]
    coords = []
    if vframe.lower() == 'edv':
        coords = d_coords
        _overlay = edv
    elif vframe.lower() == 'esv':
        coords = s_coords
        _overlay = esv
    # Create polygon mask and image
    dmask_fig, ax = plt.subplots(figsize=(f_len, f_len),
                                 dpi=1,
                                 frameon=False,
                                 tight_layout=True)
    ax.axis('off')
    # seperate segments left hand and right hand coordinate pairs
    lpoints, rpoints = [], []
    for coord in coords:
        lpoints.append((float(coord[0]), float(coord[1])))
        rpoints.append((float(coord[2]), float(coord[3])))
    # reverse right hand pairs of coordinates for concatenation
    rpoints = rpoints[::-1]
    # remove the last right hand point, this is the bottom of the LV length,...
    # to satisfy closed polygon requirement the last point must be the top...
    # of the LV length
    rpoints = rpoints[0:-1]
    # closepoint of polygon is equal to first
    closepoint = [lpoints[0]]
    # create base, original video frames are 112, 112
    base = np.zeros((112, 112), dtype=np.uint8)
    # concatenate points into required order for Polygon object
    poly_points = np.concatenate((lpoints, rpoints, closepoint))
    # check that the last point is the same as the first point
    assert (poly_points[0][0] == poly_points[-1][0])
    assert (poly_points[0][1] == poly_points[-1][1])
    poly = plt.Polygon(poly_points, color='w')
    prefix = vframe + '_'
    if not overlay:
        plt.imshow(base, interpolation='none', cmap='gray')  # mask
    if overlay:
        plt.imshow(_overlay, interpolation='none', cmap='gray')  # overlay
        prefix = prefix + 'overlay_'
    ax.add_patch(poly)
    dmask_fig.savefig(prefix + name[0] + '.png')

    plt.close(dmask_fig)


# generate mask files for training set
def gen_mask_images(path, tdata, vframe='edv', overlay=False):
    path = str(path)
    for file in os.listdir(path):
        print(file)
        _video, _name = load_video(path, str(file))
        gen_mask(_video, _name, tdata, vframe, overlay)


fl = load_filelist_data(file_list_path)
tl = load_tracing_data(label_trace_path, fl)
gen_mask_images(val_path, tl, overlay=True)


# Function to iterate through video file path...
# and generate original echocardiogram images of...
# EDV or ESV frame
def gen_cardio_images(path, tdata, edv_esv):
    # c = 0
    path = str(path)
    for file in os.listdir(path):
        # if c == 7000:
        # break
        path = str(path)
        print(file)
        _video, _name = load_video(path, str(file))
        gen_original(_video, _name, tdata, rotation=2)
        # c += 1


# func: generate a dictionary containing
#       video names and their corresponding
#       tracing labels
# args: string -> path
#       int -> limit, if set to zero entire dataset is committed to dictionary
# ret:  Dictionary -> tracings
def gen_tracing_dict(path, limit=10):
    counter = 0
    tracings = {}
    fl = load_filelist_data(file_list_path)
    tracings_arr = load_tracing_data(path, fl)
    with open(path, newline='') as csvfile:
        header = csvfile.readline().strip().split(",")
        assert header == ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]
        print("Generating tracings dict...")
        filereader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in filereader:
            split_row = row[0].split(',')
            # init video names as dictionary keys
            if split_row[0] not in tracings:
                tracings[split_row[0]] = []
    s_time = time.perf_counter()
    for key in tracings:
        if counter == limit:
            # print("Trace generation count limit met.")
            break
        trace_temp = []
        for el in tracings_arr:
            if str(key) == el[0]:
                trace_temp.append(el[1:])
        tracings[key] = trace_temp
        counter += 1
        # print("Video: {}".format(counter))
    e_time = time.perf_counter()
    """
    ### DEBUG
    #print("test_dict_el0: ", tracings['0X100009310A3BD7FC.avi'])
    #print("test_dict_el1: ", tracings['0X1002E8FBACD08477.avi'])
    #print("test_dict_el2: ", tracings['0X1005D03EED19C65B.avi'])
    #print("test_dict_el3: ", tracings['0X10075961BC11C88E.avi'])
    """
    tracedict_time = (e_time - s_time)
    print("Trace dict time: {}".format(tracedict_time))
    return tracings


def gen_filelist_dict(path, limit=10):
    counter = 0
    filelist = {}
    with open(path, newline='') as csvfile:
        header = csvfile.readline().strip().split(",")
        assert header == ["FileName", "EF", "ESV", "EDV",
                          "FrameHeight", "FrameWidth", "FPS",
                          "NumberOfFrames", "Split"]
        print("Generating filelist dict...")
        filereader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        s_time = time.perf_counter()
        for row in filereader:
            if counter == limit:
                print("File list generation count limit met.")
                break
            split_row = row[0].split(',')
            filelist[split_row[0]] = [split_row[1:]]
            # counter += 1
            # print("Video: {}".format(counter))
        e_time = time.perf_counter()
    filelistdict_time = (e_time - s_time)
    print("File list dict time: {}".format(filelistdict_time))
    return filelist


# func: determine black/white pixel ratio of image
# args: String, path to image
# args: Tuple, height and width of image
# ret:  Int, number of background pixels
# ret:  Int, number of mask pixels
# ret:  Float, percentage of mask pixels
def mask_pixelratio(image, imgsize):
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    img = img_to_array(img)
    background = 0
    mask = 0
    for row in img:
        for pixel in row:
            if pixel == 0.0:
                background += 1
            elif pixel > 0.0:
                mask += 1

    ratio = (mask / (imgsize[0] * imgsize[1])) * 100
    return background, mask, ratio


def show_pixel_value(image):
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (16, 16))
    img = img_to_array(img)
    img[img > 0] = 255
    img = img / 255
    img = img.astype(np.int32)
    #print(img)


mask_train_dir_all = echonet_dir / 'Images' / 'Train' / 'AllMasks'
sample_paths = sorted(
    [
        os.path.join(mask_train_dir_all, fname)
        for fname in os.listdir(mask_train_dir_all)
        if fname.endswith('.png') and not fname.startswith('.')
    ]
)
sample = sample_paths[0]
w, b, r = mask_pixelratio(sample, (112, 112))
print(w, b, r)
show_pixel_value(sample)


# Parse and display predicitons
def parse_preds(preds):
    """ Convert probabilities into object and background pixel values based on a threshold. """
    _preds = (preds > 0.5).astype(int)
    _preds = _preds * 255

    return _preds


# edge detection on predicted masks
def detect_edges(i, preds, color_tuple):
    """
    function that uses numpy's implementation of the Canny edge detection
    algorithm to detect the borders of the mask
    :param i:
    :param preds:
    :return:
    """
    preds = parse_preds(preds)
    mask = preds[i]
    mask = np.array(mask).astype(np.uint8)
    mask_edges = cv2.Canny(mask, 100, 200)
    rgb_edges = cv2.cvtColor(mask_edges, cv2.COLOR_GRAY2RGB)
    #mask_edges = np.expand_dims(mask_edges, 2)
    rgb_edges *= np.array(color_tuple, np.uint8)

    return rgb_edges

def overlay_edges(underlay, edge1, edge2):
    print(underlay.shape)
    #print(edge1.shape)
    #overlay = underlay + edge1 + edge2

    overlay = cv2.addWeighted(underlay, 1, edge1, 0.5, 0)
    overlay = cv2.addWeighted(overlay, 1, edge2, 0.5, 0)

    #plt.imshow(underlay)
    #plt.imshow(edge1)
    #plt.imshow(edge2)
    #plt.imshow(ed_mask)
    #plt.show()
    print(underlay.shape)

    return overlay



