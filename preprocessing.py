#matplotlib inline

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom
import os
import scipy.ndimage
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import time
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Some constants
INPUT_FOLDER = 'input/stage1/'
MIN_BOUND = -1000.0
MAX_BOUND = 400.0
PIXEL_MEAN = 0.25
patients = os.listdir(INPUT_FOLDER)
patients.sort()
labels = pd.read_csv('input/stage1_labels.csv',index_col=0)

# Load the scans in given folder path
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices


def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def resample(image, scan, new_spacing=[1, 1, 1]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing

def resize(image, scan, size=[128, 128, 128]):
    real_resize_factor = np.array(size,dtype=float) / image.shape
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    return image

def plot_3d(image, threshold=-300):
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2, 1, 0)

    verts, faces = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()
def save_3d(image, path, threshold=-300):
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2, 1, 0)

    verts, faces = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.savefig(path)

def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None


def segment_lung_mask(image, fill_lung_structures=True):
    # not actually binary, but 1 and 2.
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8) + 1
    labels = measure.label(binary_image)

    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air
    #   around the person in half
    background_label = labels[0, 0, 0]

    # Fill the air around the person
    binary_image[background_label == labels] = 2

    # Method of filling the lung structures (that is superior to something like
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)

            if l_max is not None:  # This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    binary_image -= 1  # Make the image actual binary
    binary_image = 1 - binary_image  # Invert it, lungs are now 1

    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None:  # There are air pockets
        binary_image[labels != l_max] = 0

    return binary_image
def segment_lung_mask_linear(image):
    # not actually binary, but 1 and 2.
    # 0 is treated as background, which we do not want
	image=image+600-250
	image[image > 700]=700
	image[image < 0]=0;
	image=image.astype(float)
	image=image*255/700
	image=image.astype(np.uint8)
	return image
def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

def zero_center(image):
    image = image - PIXEL_MEAN
    return image

def process_data(temp_patient):
    each_patient = load_scan(INPUT_FOLDER + temp_patient)
    each_patient_pixels = get_pixels_hu(each_patient)
    #pix_resampled, spacing = resample(each_patient_pixels, each_patient, [1, 1, 1])
    #segmented_lungs_fill = segment_lung_mask(pix_resampled, True)
    pix_resampled = resize(each_patient_pixels, each_patient)
    segmented_lungs_fill = segment_lung_mask_linear(pix_resampled)
    #segmented_lungs_fill.resize(128, 256, 256)
    return segmented_lungs_fill

def second_process_data(patitent,df_label, has_label = True):
    if has_label:
        label = df_label.get_value(patient, 'cancer')
        if label == 1: label=np.array([0,1])
        elif label == 0: label=np.array([1,0])
        return process_data(patient),label
    else:
        return process_data(patient)

if __name__ == "__main__":
    ticks = time.time()
    much_data = []
    no_label = []
    no_label_data = []
    for num, patient in enumerate(patients):
        try:
            img_data,label = second_process_data(patient,labels,True)
            #save_3d(img_data,path='img/'+patient+'.jpg',threshold=0)
            much_data.append([img_data,label])
            print (num)
        except KeyError as e:
            print (num, "This is a key error")
            no_label.append(patient)
        print ("count ",num,":\t",time.time()-ticks)
    for no_label_patient in no_label:
        no_label_data.append(second_process_data(no_label_patient,labels,False))
    np.save('muchdata-128-128-128-validation.npy',much_data[0:200])#validation data
    np.save('muchdata-128-128-128-train.npy',much_data[200:])#train data
    np.save('nolabeldata-128-128-128.npy',no_label_data)#no label data