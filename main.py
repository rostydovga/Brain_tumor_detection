from glob import glob
from os.path import join
import os
import cv2
import zipfile
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from skimage.feature import hog, local_binary_pattern
from tqdm import tqdm
import sklearn
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten
from keras.models import Model
from tensorflow.keras.layers import Conv1D, Flatten, Dense


ARCHIVE = 'archive.zip'
TRAINING = 'Training'
TESTING = 'Testing'
LABEL_LOC = 1 # is the location of the class label in the image path 
IMAGE_SIZE = (224,224)
GPU_AVAILABLE = False

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))
if len(physical_devices)>1:
    GPU_AVAILABLE=True


'''
'''
def classes_name(path):
    return [f.split('\\')[-1] for f in glob(join(path,'*'))]

'''
'''
def get_label_image(image):
  return labels_dictionary[image.split('\\')[LABEL_LOC]]

'''
'''
def extract_features(images, feat_type, img_size=None):

    labels = []
    features = []

    for image in tqdm(images):

        img = cv2.imread(image, 0)

        if img_size != None:
          img = cv2.resize(img, (img_size, img_size))

        if feat_type == 'hog':
            feat = hog(img, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(1, 1))
        elif feat_type == 'lbp':
            feat = np.ravel(local_binary_pattern(img, P=100, R=5))
        elif feat_type == 'img':
            img = img / 256.0
            feat = np.ravel(img)
        else:
            raise NotImplementedError('Not implemented feature!')

        features.append(feat)
        labels.append(get_label_image(image))

    return features, labels

'''
'''
def create_validation_set(training_set, perc=0.1):
    split_point = 1-perc

    if split_point>1:
        raise ValueError
    else:
        training_images = training_set[:int(0.9*len(training_set))]
        validation_images = training_set[int(0.9*len(training_set)):]
        return training_images, validation_images
    
def ml_classification_approach(training_set, validation_set, testing_set):
    # create the classifiers
    clf_svc = svm.SVC(gamma=0.001, C=100., kernel='rbf', verbose=False)
    clf_rand = RandomForestClassifier(max_depth=2, random_state=0)
    clf_ada = AdaBoostClassifier(n_estimators=100, random_state=True)
    clf_tree = DecisionTreeClassifier()


    # variables
    dictionary_results = {}
    img_size = 64
    classifiers = [clf_svc, clf_rand, clf_ada, clf_tree]
    feature_types = ['hog','lbp','img']

    # for each type of feature extraction
    for feature in feature_types:


        dictionary_results_classifiers = {}

        # subdivide the datasets
        train_x, train_y = extract_features(training_set, feature, img_size)
        val_x, val_y = extract_features(validation_set, feature, img_size)
        test_x, test_y = extract_features(testing_set, feature, img_size)

        print(f'Feature type: {feature}')

        for clf in classifiers:

            dictionary_accuracy = {}

            print(f'-- Train {clf} --')
            clf.fit(train_x, train_y)
            # score
            score = clf.score(val_x, val_y)
            # accuracy
            y_pred = clf.predict(test_x)
            accuracy = accuracy_score(test_y, y_pred)
            # matrix
            matrix = confusion_matrix(test_y, y_pred)
            #print(f'Matrix diagonal --> {matrix.diagonal() / matrix.sum(axis=1)}')
            # cm = ConfusionMatrixDisplay(matrix)
            # cm.plot()
            # add measures to the dictionary
            dictionary_accuracy['score'] = score
            dictionary_accuracy['accuracy'] = accuracy
            #dictionary_accuracy['matrix'] = matrix

            dictionary_results_classifiers[clf] = dictionary_accuracy


        dictionary_results[feature] = dictionary_results_classifiers
        print()
    return dictionary_results


'''
'''
def create_train_val_test_set_keras(validation_split=0.2):
    class_name = classes_name(TRAINING)

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory=TRAINING,
        labels='inferred', # where are the labels, 'inferred' means that labels are automatically created by the method
        label_mode='categorical',
        class_names=class_name,
        color_mode='grayscale',
        batch_size=32,
        image_size=IMAGE_SIZE,
        shuffle=True,
        seed=1307,
        validation_split=validation_split,
        subset='training', 
        interpolation='bilinear',
        follow_links=False,
        crop_to_aspect_ratio=False,
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory=TRAINING,
        labels='inferred', # where are the labels, 'inferred' means that labels are automatically created by the method
        label_mode='categorical',
        class_names=class_name,
        color_mode='grayscale',
        batch_size=32,
        image_size=IMAGE_SIZE,
        shuffle=True,
        seed=1307,
        validation_split=validation_split,
        subset='validation', # I want to create a training set that is 80% of the dataset, I define that the validation has 20% of the dataset
        interpolation='bilinear',
        follow_links=False,
        crop_to_aspect_ratio=False,
    )

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory=TESTING,
        labels='inferred', # where are the labels, 'inferred' means that labels are automatically created by the method
        label_mode='categorical',
        class_names=class_name,
        color_mode='grayscale',
        batch_size=32,
        image_size=IMAGE_SIZE,
        shuffle=True,
        seed=1307,
        validation_split=0,
        interpolation='bilinear',
        follow_links=False,
        crop_to_aspect_ratio=False,
    )

    return train_ds, val_ds, test_ds

def model_no_conv_layer():
    model=keras.models.Sequential([
        tf.keras.layers.Normalization(axis=-1, mean=None, variance=None),
        keras.layers.Flatten(input_shape=(64, 64,)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(4, activation='softmax')
    ])

    return model

def model_with_conv_layer():
    model=keras.models.Sequential([
        tf.keras.layers.Normalization(axis=-1, mean=None, variance=None),
        Conv1D(64,8, activation='relu'), # 64,60,64
        Conv1D(32,4, activation='relu'), # 64, 54, 32
        Flatten(input_shape=[64,54,32]),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(4, activation='softmax')
    ])

    return model

def model_mobile_net_tuned():
    model_MobileNet = tf.keras.applications.MobileNet(
        input_shape=None,
        alpha=1.0,
        depth_multiplier=1,
        dropout=0.001,
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        pooling=None,
        classes=1000,
        classifier_activation="softmax"
    )

    # Create a layer where input is the output of the second last layer
    output = Dense(4, activation='softmax', name='predictions')(model_MobileNet.layers[-2].output)

    # Then create the corresponding model
    model_MobileNet = Model(model_MobileNet.input, output)

    return model_MobileNet


def train_model(model, train_ds, val_ds, epochs=20):

    # function used during the training
    callbacks = [
        # to save the model after every epoch
        keras.callbacks.ModelCheckpoint("checkpoints/save_at_{epoch}.h5"), # save the weights and biasis
        # logging -> able to memorize all the logs produced dutìring the training
        tf.keras.callbacks.TensorBoard(log_dir="logs", write_graph=True, write_images=False, update_freq="epoch",)
    ]

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy", # loss function we're gonna use for classification task
        metrics=["accuracy"],
    )

    model.fit(
        train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
    )

    return model

def evaluate_model(model, test_ds):
    return model.evaluate(
                x=test_ds,
                y=None,
                batch_size=32,
                verbose=True,
                sample_weight=None,
                steps=None,
                callbacks=None,
                max_queue_size=10,
                workers=1,
                use_multiprocessing=False,
                return_dict=False,
            )




'''
'''
def main():

    '''
        MACHINE LEARNING APPROACH
        can be done in the local machine without the need of a GPU
    '''
    file_name_results_ML = 'ml_classification_results.csv'

    # extract archive.zip
    if TESTING and TRAINING not in os.listdir():
        print('extract the archive...')
        with zipfile.ZipFile(ARCHIVE, 'r') as zip:
            zip.extractall()

    # for each class have the path for the images
    training_images = glob(join(TRAINING,'*','*.jpg'))
    testing_images = glob(join(TESTING,'*','*.jpg'))

    classes = classes_name(TRAINING)
    print(f'The classes are: {classes}')

    
    global labels_dictionary
    labels_dictionary = {}
    for num, val in enumerate(classes):
        labels_dictionary[val] = num


    # extract keys and values
    key_list = list(labels_dictionary.keys())
    val_list = list(labels_dictionary.values())

    print('Shuffle training set and testing set...')
    # shuffle the images
    np.random.shuffle(training_images)
    np.random.shuffle(testing_images)
    
    print('Create a validation set from the training set...')
    # create a validation set from the training set
    training_images, validation_images = create_validation_set(training_images)
    print(f'{len(training_images)=}, {len(validation_images)=}, {len(testing_images)=}')

    print('ML approach...')
    dictionary_result = ml_classification_approach(training_images, validation_images, testing_images)

    print(f'The results of the Machine Learning Classification approach are saved in {file_name_results_ML}')
    pd.DataFrame(dictionary_result).to_csv(file_name_results_ML)

    '''
        DEEP LEARNING APPROACH
        need a GPU
    '''
    


    # creation of training, validation and testing
    train_ds, val_ds, test_ds = create_train_val_test_set_keras()

    # SIMPLE MODEL
    print('Create model with no Conv layer...')
    model_simple = model_no_conv_layer()
    print(model_simple.summary())

    if GPU_AVAILABLE:
        print('Training simple model...')
        model_simple_trained = train_model(model_simple, train_ds, val_ds)
        print(evaluate_model(model_simple_trained, test_ds))

    # MODEL WITH CONV LAYERS
    print('Create model with Conv layers....')
    model_conv = model_with_conv_layer()
    print(model_conv.summary())

    if GPU_AVAILABLE:
        print('Training Conv model...')
        model_conv_trained = train_model(model_conv, train_ds, val_ds)
        print(evaluate_model(model_conv_trained, test_ds))

    # TRANSFER LEARNING
    print('Tune MobileNet model for fine tuning...')
    model_mobile_net = model_mobile_net_tuned()
    print(model_mobile_net.summary())

    if GPU_AVAILABLE:
        print('Training MobielNet tuned...')
        model_mobile_net_trained = train_model(model_mobile_net, train_ds, val_ds)
        print(evaluate_model(model_mobile_net_trained, test_ds))


        





if __name__ == "__main__":
    main()

    print('FINISH....')

