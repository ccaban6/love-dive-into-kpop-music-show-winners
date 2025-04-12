import essentia
from essentia.standard import (
    TensorflowPredictVGGish,
    TensorflowPredictEffnetDiscogs,
    TensorflowPredict2D,
    TempoCNN,
    MonoLoader,
    Resample
)
import numpy as np
import logging
import os
import time  # Add import for timing
import tensorflow as tf  # Import TensorFlow for GPU memory management
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs

class EssentiaPredictor:
    def __init__(self):
        # Initialize logger
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.logger.info("Initializing models in batches...")
        try:
            # Define model configurations
            model_configs = [
                {'name': 'vggish_embedding_model', 'type': 'vggish', 'graph': "Models/audioset-vggish-3.pb", 'output': "model/vggish/embeddings"},
                {'name': 'effnet_embedding_model', 'type': 'effnet', 'graph': "Models/discogs-effnet-bs64-1.pb", 'output': "PartitionedCall:1"},
                {'name': 'vggish_dance_model', 'type': '2d', 'graph': "Models/danceability-audioset-vggish-1.pb", 'output': "model/Softmax"},
                {'name': 'vggish_party_model', 'type': '2d', 'graph': "Models/mood_party-audioset-vggish-1.pb", 'output': "model/Softmax"},

                {'name': 'vggish_happy_model', 'type': '2d', 'graph': "Models/mood_happy-audioset-vggish-1.pb", 'output': "model/Softmax"},
                {'name': 'vggish_sad_model', 'type': '2d', 'graph': "Models/mood_sad-audioset-vggish-1.pb", 'output': "model/Softmax"},
                {'name': 'effnet_party_model', 'type': '2d', 'graph': "Models/mood_party-discogs-effnet-1.pb", 'output': "model/Softmax"},
                {'name': 'effnet_happy_model', 'type': '2d', 'graph': "Models/mood_happy-discogs-effnet-1.pb", 'output': "model/Softmax"},

                {'name': 'effnet_sad_model', 'type': '2d', 'graph': "Models/mood_sad-discogs-effnet-1.pb", 'output': "model/Softmax"},
                {'name': 'effnet_approachability_model', 'type': '2d', 'graph': "Models/approachability_2c-discogs-effnet-1.pb", 'output': "model/Softmax"},
                {'name': 'effnet_engagement_model', 'type': '2d', 'graph': "Models/engagement_2c-discogs-effnet-1.pb", 'output': "model/Softmax"},
                {'name': 'effnet_timbre_model', 'type': '2d', 'graph': "Models/timbre-discogs-effnet-1.pb", 'output': "model/Softmax"},

                {'name': 'effnet_genre_model', 'type': '2d', 'graph': "Models/genre_discogs400-discogs-effnet-1.pb", 'output': "PartitionedCall:0", 'input': "serving_default_model_Placeholder"},
                {'name': 'tempo_model', 'type': 'tempo', 'graph': 'Models/deeptemp-k16-3.pb', 'output': ""}
            ]

            # Load models
            for i in range(0, len(model_configs)):
                config = model_configs[i]
                self.logger.info(f"Initializing model {i + 1}: {config['name']}...")
                start_time = time.time()  # Start timing

                setattr(self, config['name'], self.create_model(
                    model_type=config['type'],
                    graph_path=config['graph'],
                    output=config.get('output', ""),
                    input_val=config.get('input', "model/Placeholder")
                ))

                # Log GPU memory usage
                gpu_devices = tf.config.experimental.list_physical_devices('GPU')
                if gpu_devices:
                    for gpu in gpu_devices:
                        memory_info = tf.config.experimental.get_memory_info('GPU:0')
                        self.logger.info(f"GPU memory usage after loading {config['name']}: "
                                         f"Used: {memory_info['current']} bytes, "
                                         f"Peak: {memory_info['peak']} bytes")

                # Clear GPU memory
                tf.keras.backend.clear_session()

                end_time = time.time()  # End timing
                elapsed_time = end_time - start_time
                self.logger.info(f"Model {i + 1} ({config['name']}) initialized in {elapsed_time:.2f} seconds.")

        except Exception as e:
            self.logger.error(f"Error during model initialization: {e}")
            raise


    @staticmethod
    def create_model(model_type, graph_path, output, input_val):
        try:
            if model_type == 'vggish':
                return TensorflowPredictVGGish(graphFilename=graph_path, output=output)
            elif model_type == 'effnet':
                return TensorflowPredictEffnetDiscogs(graphFilename=graph_path, output=output)
            elif model_type == '2d':
                return TensorflowPredict2D(graphFilename=graph_path, output=output, input=input_val)
            elif model_type == 'tempo':
                return TempoCNN(graphFilename=graph_path)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        except Exception as e:
            logging.error(f"Error loading model {model_type} from {graph_path}: {e}")
            raise

    def predict_all(self, audio_file):
        """
        Processes an audio file to return predictions from multiple models,
        excluding the genre model which doesn't return a numeric value.
        
        Parameters:
          audio_file (str): Path to the audio file.
          
        Returns:
          dict: A dictionary with keys for each model and the corresponding prediction values.
        """
        # Load the audio once using MonoLoader
        audio = MonoLoader(filename=audio_file, sampleRate=16000, resampleQuality=0)()
        predictions = {}

        # Process VGGish-based predictions:
        vggish_embedding = self.vggish_embedding_model(audio)
        predictions["vggish_dance"] = 100 * np.mean(self.vggish_dance_model(vggish_embedding)[:, 0])
        predictions["vggish_party"] = 100 * np.mean(self.vggish_party_model(vggish_embedding)[:, 0])
        predictions["vggish_happy"] = 100 * np.mean(self.vggish_happy_model(vggish_embedding)[:, 0])
        predictions["vggish_sad"]   = 100 * np.mean(self.vggish_sad_model(vggish_embedding)[:, 0])
        
        # Process Effnet-based predictions:
        effnet_embedding = self.effnet_embedding_model(audio)
        predictions["effnet_party"] = 100 * np.mean(self.effnet_party_model(effnet_embedding)[:, 0])
        predictions["effnet_happy"] = 100 * np.mean(self.effnet_happy_model(effnet_embedding)[:, 0])
        predictions["effnet_sad"]   = 100 * np.mean(self.effnet_sad_model(effnet_embedding)[:, 0])
        predictions["effnet_approachability"] = 100 * np.mean(self.effnet_approachability_model(effnet_embedding)[:, 1])
        predictions["effnet_engagement"]      = 100 * np.mean(self.effnet_engagement_model(effnet_embedding)[:, 1])
        predictions["effnet_timbre_bright"]   = 100 * np.mean(self.effnet_timbre_model(effnet_embedding)[:, 0])
        
        # Process Tempo prediction: resample the audio before prediction
        resampler = Resample(inputSampleRate=16000, outputSampleRate=11025)
        audio_resampled = resampler(audio)
        predictions["tempo"] = self.tempo_model(audio_resampled)[0]
        
        # Exclude the genre model since it doesn't return a numeric value.
        return predictions