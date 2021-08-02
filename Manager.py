import os
import medpy
import medpy.io
import numpy as np
import matplotlib.pyplot as plt
import cv2

#Plaid-ml
#import keras.applications as keraps

#Tensorflow executable
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'            #Suppress INFO and WARNING messages
import tensorflow.compat.v1.keras.applications as keraps

from NeuralNetModel import NeuralNetModel

class Manager():
    """Used to work with ct-images and neural networks"""
    def __init__(self):
        self.detection_neural_network = None        #Keras NN model used to detect cancer in a single image
        self.localization_neural_network = None     #Keras NN model used to localize a nodule in a single image

        self.ct_scan = None                         #CT scan of one patient in a form of np array
        self.ct_scan_header = None                  #CT scan header for the patient
        self.ct_scan_segmentation = None            #Segmentation data for current patient

        self.ct_scan_detection = None               #Reshaped ct images so they can be used in a detection network
        self.ct_scan_detection_rgb = False          #True if detection dataset contatins fake RGB information

        self.ct_scan_localization = None            #Reshaped ct images so they can be used in a localization network
        self.ct_scan_localization_rgb = False       #True if localization dataset contatins fake RGB information

        self.detection_results = None               #Detection neural net output    Format: ((images), (results))
        self.localization_results = None            #Localization neutal net output Format: (image, (results))

        self.lung_hounsfield_limits = (-1000, 0)    #Limits which tissue densities are most visible on ct scans
        
        self.config_file_path = './settings.cfg'    #Config file path
        
        self.debug_output = (self.read_config_file('debug') == 'True') #Toggling on debug messages according to config file
               

    def load_patient_data(self, path):
        """Loads ct scan images and their header into memory, does slight post-processing on the input image to make lungs more visible"""
        #Clearing old data in a case that user wants to load another patient
        if (self.ct_scan_detection is not None or self.ct_scan_localization is not None):
            self.ct_scan_segmentation = None
            
            self.ct_scan_detection = None
            self.ct_scan_detection_rgb = False

            self.ct_scan_localization = None
            self.ct_scan_localization_rgb = False

            self.detection_results = None
            self.localization_results = None        
            self.debug("**Previous patient data found, clearing out memory.")
        
        #Loading
        self.ct_scan, self.ct_scan_header = medpy.io.load(path)
        #Post-processing
        self.ct_scan[self.ct_scan <= -2000] = self.lung_hounsfield_limits[0]
        self.ct_scan = self.ct_scan.astype(np.float32)
        self.ct_scan = np.clip(self.ct_scan, self.lung_hounsfield_limits[0], self.lung_hounsfield_limits[1])
        #Transposing data from (x, y, z) to (z, y, x)
        self.ct_scan = np.transpose(self.ct_scan, (2, 1, 0))
        self.debug("**Patient data loaded into memory! Shape: {}".format(self.ct_scan.shape))


    def load_segmentation_data(self, path):
        """Loads segmentation data into memory"""
        #Loading
        self.ct_scan_segmentation, header = medpy.io.load(path)
        self.ct_scan_segmentation = self.ct_scan_segmentation.astype(np.int16)
        
        #Post-processing
        self.ct_scan_segmentation[self.ct_scan_segmentation >= 1] = 1
        
        #Transposing
        self.ct_scan_segmentation = np.transpose(self.ct_scan_segmentation, (2, 1, 0))
        self.debug("**Patient segmentation data loaded into memory! Shape: {}".format(self.ct_scan_segmentation.shape))
        
        if (self.ct_scan is None):
            self.debug("[!]Loading segmentation data before patient data!")
            return
        if (self.ct_scan_segmentation.shape[0] != self.ct_scan.shape[0]):
            self.ct_scan_segmentation = None
            raise Exception("Image and segmentation data not matching! Image: {} slices Segmentation: {} slices".format(self.ct_scan_segmentation.shape[0], self.ct_scan.shape[0]))


    def load_detection_network(self, path):
        self.detection_neural_network = NeuralNetModel(path)
        if (self.debug_output):
            self.detection_neural_network.debug_output = True
        

    def load_localization_network(self, path):
        self.localization_neural_network = NeuralNetModel(path)
        if (self.debug_output):
            self.localization_neural_network.debug_output = True


    def reshape_for_prediction(self, segmentation_enabled, preprocess = 'default', network = 'detection'):
        """Reshapes the CT scan images so they have a correct shape to be used in the detection model"""
        if segmentation_enabled and not self.is_segmentation_loaded():
            self.debug("[!]Segmentation enabled but no segmentation data loaded!")
            return

        image_mode = None
        if (network == 'detection'):
            image_mode = (self.get_network_input_shape(0)[3] == 3)
        elif network == 'localization':
            image_mode = (self.get_network_input_shape(1)[3] == 3)
        else:
            raise Exception("Wrong network type!")

        #False - unchecked, True - checked
        images = np.copy(self.ct_scan)
        resized_images = []
        for i in range(0, images.shape[0]):
            #Image segmentation if enabled
            if segmentation_enabled:
                #Creating the segmentation mask
                inverted_mask = np.full_like(self.ct_scan_segmentation[i], 0)
                indices_zero = self.ct_scan_segmentation[i] == 0
                inverted_mask[indices_zero] = -1000
                #Appending the mask to an image
                images[i] = images[i] * self.ct_scan_segmentation[i]
                images[i] = images[i] + inverted_mask

            #Normalizing the image
            if (np.max(images[i]) != -1000):
                if (image_mode):
                    images[i] = ((images[i] + abs(np.min(images[i]))) / (np.max(images[i]) + abs(np.min(images[i])))) * 255.0
                    images[i] = np.rint(images[i])
                else:
                    images[i] = ((images[i] + abs(np.min(images[i]))) / (np.max(images[i]) + abs(np.min(images[i]))))
            else:
                images[i] = 0
        
            if (image_mode):
                images[i] = images[i].astype(np.int32)
            else:
                images[i] = images[i].astype(np.float32)

            #Resizing the image
            resolution = self.get_network_input_shape(0)[1]
            resized_image = cv2.resize(images[i], (resolution, resolution))
            resized_images.append(resized_image)
            
        if network == 'detection':
            self.ct_scan_detection = np.array(resized_images)
            network_shape = self.get_network_input_shape(0)
            if image_mode:
                self.ct_scan_detection = np.repeat(self.ct_scan_detection[..., np.newaxis], 3, -1)
                self.ct_scan_detection = np.reshape(self.ct_scan_detection, (self.ct_scan_detection.shape[0], network_shape[1], network_shape[1], 3))
                self.ct_scan_detection_rgb = True
            else:
                self.ct_scan_detection = np.reshape(self.ct_scan_detection, (self.ct_scan_detection.shape[0], network_shape[1], network_shape[1], 1))
                self.ct_scan_detection_rgb = False
            self.ct_scan_detection_reshaped = True
            self.debug("Detection network dataset shape: {}".format(self.ct_scan_detection.shape))
        
        elif network == 'localization':
            self.ct_scan_localization = np.array(resized_images)
            network_shape = self.get_network_input_shape(1)
            if image_mode:
                self.ct_scan_localization = np.repeat(self.ct_scan_localization[..., np.newaxis], 3, -1)
                self.ct_scan_localization = np.reshape(self.ct_scan_localization, (self.ct_scan_localization.shape[0], network_shape[1], network_shape[1], 3))
                self.ct_scan_localization_rgb = True
            else:
                self.ct_scan_localization = np.reshape(self.ct_scan_localization, (self.ct_scan_localization.shape[0], network_shape[1], network_shape[1], 1))
                self.ct_scan_localization_rgb = False
            self.ct_scan_localization_reshaped = True
            self.debug("Localization network dataset shape: {}".format(self.ct_scan_localization.shape))

        #Applies the keras pre-process to a selected dataset in a case that it is needed.
        if (preprocess != 'default'):
            self.keras_preprocess_dataset(network, preprocess)

    
    def keras_preprocess_dataset(self, network, preprocess):
        """Applies the keras preprocess if needed"""
        try:
            if network == 'detection':
                if preprocess == 'vgg19':
                    self.ct_scan_detection = keraps.vgg19.preprocess_input(self.ct_scan_detection)
                elif preprocess == 'resnet50':
                    self.ct_scan_detection = keraps.resnet50.preprocess_input(self.ct_scan_detection)
                elif preprocess == 'densenet169':
                    self.ct_scan_detection = keraps.densenet.preprocess_input(self.ct_scan_detection)
            elif network == 'localization':
                if preprocess == 'vgg19':
                    self.ct_scan_localization = keraps.vgg19.preprocess_input(self.ct_scan_localization)
                elif preprocess == 'resnet50':
                    self.ct_scan_localization = keraps.resnet50.preprocess_input(self.ct_scan_localization)
                elif preprocess == 'densenet169':
                    self.ct_scan_localization = keraps.densenet.preprocess_input(self.ct_scan_localization)
        except:
            self.debug("[!]There was an error trying to preprocess the dataset for {} specification.".format(preprocess))
            
            
    def predict_detection_model(self):
        """Sends the detection dataset to be predicted by detection neural network. Saves detection results to a variable."""
        if self.ct_scan_detection is not None:
            self.detection_results = self.detection_neural_network.predict_using_model(self.ct_scan_detection)
        else:
            self.debug("[!]No reshaped images found! Prediction cant be done!")
            raise Exception("No reshaped images found! Prediction cant be done!")


    def predict_localization_model(self):
        """Sends the localization dataset to be predicted by localization neural network. Saves the results into a variable."""
        if self.ct_scan_localization is not None:
            self.localization_results = self.localization_neural_network.predict_using_model(self.ct_scan_localization)
        else:
            self.debug("[!]No reshaped images found! Prediction cant be done!")
            raise Exception("No reshaped images found! Prediction cant be done!")


    def get_slice_Z_coordinates(self, slice_number):
        """Returns the Z coordinate of current slice"""
        origin = self.ct_scan_header.sitkimage.GetOrigin()
        spacing = self.ct_scan_header.sitkimage.GetSpacing()
        return ((slice_number * spacing[2]) + origin[2])
    

    def get_treeview_data(self):
        """Returns data used to show ct scan info in a treeview"""
        data = []
        counter = 0
        if self.detection_results is None:
            for image in self.ct_scan:
                data.append((counter, self.get_slice_Z_coordinates(counter), 'NaN%'))
                counter += 1
        else:
            for i in range(0, self.ct_scan.shape[0]):
                coordinates = self.get_slice_Z_coordinates(i)
                probability = self.detection_results[i]
                data.append((i, coordinates, probability[0]))
        return data


    def get_dataset_resolution(self):
        return self.ct_scan.shape


    def get_slice_image(self, mode = 0, slice_number = 0, resolution = (255, 255)):
        image_array = None
        if (mode == 0):
            image_array = np.copy(self.ct_scan[slice_number])
            image_array = ((image_array + abs(np.min(image_array))) / (np.max(image_array) + abs(np.min(image_array))))
            image_array = ((image_array) * 255).astype(np.uint8)
        elif (mode == 1):
            image_array = np.copy(self.ct_scan_detection[slice_number])
            if not self.ct_scan_detection_rgb:
                if (np.max(image_array) != 0.0):
                    image_array = ((image_array + abs(np.min(image_array))) / (np.max(image_array) + abs(np.min(image_array))))
                    image_array = ((image_array) * 255).astype(np.uint8)
                else:
                    image_array = 0
            else:
                if (np.max(image_array) != 0):
                    image_array = (image_array).astype(np.uint8)
                else:
                    image_array = 0
        elif (mode == 2):
            image_array = np.copy(self.ct_scan_localization[slice_number])
        else:
            raise Exception("Mode not implemented!")
        
        if resolution == 0:
            return image_array
        return cv2.resize(image_array, resolution)


    def write_config_file(self, setting, mode):
        file = open(self.config_file_path, 'r')
        config_dict = {}
        for line in file:
            split_line = line.rstrip('\n').split(':')
            config_dict[split_line[0]] = split_line[1]
        file.close()
        config_dict[setting] = mode

        file = open(self.config_file_path, 'w')
        for key, value in config_dict.items():
            file.write('{}:{}\n'.format(key, value))
        file.close()
    
    def read_config_file(self, setting):
        if not os.path.isfile(self.config_file_path):
            file = open(self.config_file_path, 'w')
            file.write("localization:en\n")
            file.write("debug:False\n")
            file.close()
            return self.read_config_file(setting)
        else:
            file = open(self.config_file_path, 'r')
            for line in file:
                line = line.rstrip('\n')
                split_line = line.split(':')
                if split_line[0] == setting:
                    return split_line[1]
            file.close()
        raise Exception("Invalid or non-existent config setting!")

    def get_network_input_shape(self, network):
        """Return the input shape of the first layer of selected network"""
        #0 - Detection network, 1 - Localization network
        if (network == 0):
            if len(self.detection_neural_network.input_shape) == 1:
                return self.detection_neural_network.input_shape[0]
            return self.detection_neural_network.input_shape
        elif (network == 1):
            if len(self.localization_neural_network.input_shape) == 1:
                return self.localization_neural_network.input_shape[0]
            return self.localization_neural_network.input_shape
        else:
            raise Exception("Unsupported input!")

    def get_localization(self):
        return self.read_config_file('localization')

    def set_localization(self, language):
        self.write_config_file('localization', language)
        
    def is_detection_network_loaded(self):
        return self.detection_neural_network is not None

    def is_localization_network_loaded(self):
        return self.localization_neural_network is not None
        
    def is_dataset_loaded(self):
        return self.ct_scan is not None
        
    def is_segmentation_loaded(self):
        return self.ct_scan_segmentation is not None
    
    def is_detection_dataset_reshaped(self):
        return self.ct_scan_detection is not None

    def is_localization_dataset_reshaped(self):
        return self.ct_scan_localization is not None

    def debug(self, message):
        if (self.debug_output):
            print(message)