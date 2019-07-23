import tqdm
from scipy.spatial import distance
from skimage.transform import resize
import tensorflow as tf
import numpy as np
from pathlib import Path
import imageio
from src.facenet.facenet.src import facenet
from src.facenet.facenet.src.align import detect_face

MODEL_DIR = str(Path('models/20180402-114759/').absolute())

def get_embeddings(image_paths, image_size=160,margin=44):  
    """ Get embeddings by class
    Args:
        image_paths - Paths to images
        image_size = Imageize (height, width) in pixels.
        margin - Margin for the crop around the bounding box (height, width) in pixels.
    Returns:
        Numpy array of images x embeddings, and a label array
    """
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Load the model
            facenet.load_model(MODEL_DIR)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Get images for the batch
            images, corresponding_paths = load_and_align_data(image_paths, image_size, margin)

            print('Calculating embeddings...')
            # Use the facenet model to calcualte embeddings
            embed = sess.run(embeddings, 
                             feed_dict={
                                 images_placeholder: images, 
                                 phase_train_placeholder:False}
                            )

        return embed, images, corresponding_paths

def crop_face(img, bounding_box, margin=44, target_size=160):
    """ Crop, resize and prewhiten face """
    current_size = np.asarray(img.shape)[0:2]
    det = np.squeeze(bounding_box[0:4])
    bb = np.zeros(4, dtype=np.int32)
    bb[0] = np.maximum(det[0]-margin/2, 0)
    bb[1] = np.maximum(det[1]-margin/2, 0)
    bb[2] = np.minimum(det[2]+margin/2, current_size[1])
    bb[3] = np.minimum(det[3]+margin/2, current_size[0])
    cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
    aligned = resize(cropped, (target_size, target_size), mode='constant', anti_aliasing=False)
    
    return facenet.prewhiten(aligned)

def _load_detect_nets():
    """ Load MTCNN_face_detection_alignment """
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            nets = detect_face.create_mtcnn(sess, None)
    return nets

def _detect_faces(img, nets, minsize=20, threshold=[0.6, 0.7, 0.7], factor=0.709):
    """ Given an image path, and an initalized network, detect faces """
    pnet, rnet, onet = nets
    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    return bounding_boxes

def load_and_align_data(image_paths, target_size, margin=None, minsize=None, threshold=None, factor=None):
    print('Creating networks and loading parameters')
    nets = _load_detect_nets()

    print('Loading, cropping, and aligning')
    corresponding_paths = []
    cropped_images = []
    for path in tqdm.tqdm_notebook(image_paths):
        img = imageio.imread(path)
        bounding_boxes = _detect_faces(img, nets)
        for bb in bounding_boxes:
            cropped_images.append(crop_face(img, bb, margin=margin))
            corresponding_paths.append(path)

    return np.stack(cropped_images), corresponding_paths