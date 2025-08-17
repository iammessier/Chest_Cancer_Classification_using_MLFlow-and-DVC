import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import tensorflow as tf

class PredictionPipeline:
    def __init__(self,filename):
        self.filename =filename
    
    def predict(self):
        ## load model
        
        model = load_model(os.path.join("artifacts","training", "model.h5"))
        # model = load_model(os.path.join("model", "model.h5"))
        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (224,224))
        test_image = image.img_to_array(test_image)
        # Apply the same preprocessing as during training (rescale to 0-1)
        test_image = test_image / 255.0
        test_image = np.expand_dims(test_image, axis = 0)
        
        # Get prediction probabilities
        prediction_probs = model.predict(test_image)
        result = np.argmax(prediction_probs, axis=1)
        
        # Print debug information
        print(f"Prediction probabilities: {prediction_probs}")
        print(f"Argmax result: {result}")
        print(f"Class index: {result[0]}")

        # Determine class mapping based on data directory structure
        # Check if the data directories exist and get their order
        data_dir = "artifacts/data_ingestion/Data"
        if os.path.exists(data_dir):
            class_dirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
            print(f"Found class directories: {class_dirs}")
            
            # Map class indices to labels
            if len(class_dirs) >= 2:
                if result[0] == 0:
                    prediction = class_dirs[0].title()  # First class
                elif result[0] == 1:
                    prediction = class_dirs[1].title()  # Second class
                else:
                    prediction = f'Unknown Class (Index: {result[0]})'
            else:
                # Fallback to hardcoded mapping if directories not found
                if result[0] == 0:
                    prediction = 'Adenocarcinoma Cancer'
                elif result[0] == 1:
                    prediction = 'Normal'
                else:
                    prediction = f'Unknown Class (Index: {result[0]})'
        else:
            # Fallback to hardcoded mapping if data directory not found
            if result[0] == 0:
                prediction = 'Adenocarcinoma Cancer'
            elif result[0] == 1:
                prediction = 'Normal'
            else:
                prediction = f'Unknown Class (Index: {result[0]})'
            
        return [{ "image" : prediction}]