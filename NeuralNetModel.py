#Plaid-ml
#from keras.models import load_model, Model

#Tensorflow executable
from tensorflow.keras.models import load_model, Model

class NeuralNetModel():
    """Class used to hold neural network models made in keras"""

    def __init__(self, model_path):
        self.path_to_model = model_path
        self.model = None
        self.input_shape = None
        self.debug_output = False

        self.load_model()

    def load_model(self):
        self.model = load_model(self.path_to_model)
        self.input_shape = self.model.layers[0].input_shape
        self.debug("**Model {} loaded!\n**Input shape: {}".format(self.path_to_model, self.input_shape))

    def predict_using_model(self, data):
        """Uses the model to predict the correct label, right data format has to be used!"""
        results = self.model.predict(data, verbose = 1)
        self.debug("**Model prediction complete!")
        return results

    def debug(self, message):
        if (self.debug_output):
            print(message)