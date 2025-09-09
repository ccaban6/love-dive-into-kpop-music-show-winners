import essentia
from essentia.standard import (
    MonoLoader, 
    TensorflowPredictEffnetDiscogs, 
    TensorflowPredict2D,
    TensorflowPredictMusiCNN, 
    TempoCNN, 
    Resample, 
    RhythmExtractor2013,
    KeyExtractor
)
import numpy as np
import json
import time
import os
import logging
import signal
import warnings
from contextlib import contextmanager



@contextmanager
def timeout(duration):
    """Context manager for timing out operations."""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {duration} seconds")
    
    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(duration))
    
    try:
        yield
    finally:
        # Restore the old signal handler
        signal.signal(signal.SIGALRM, old_handler)
        signal.alarm(0)

class EssentiaFeatureExtractor:
    def __init__(self, model_path="../essentia_models"):
        """
        Initialize the feature extractor with paths to Essentia models.
        
        Args:
            model_path (str): Path to directory containing Essentia model files
        """
        self.model_path = model_path
        print("Initializing feature extractor...")
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Model file paths
        self.model_paths = {
            # Embedding extractors
            'discogs_effnet': f"{model_path}/discogs-effnet-bs64-1.pb",
            'msd_musicnn': f"{model_path}/msd-musicnn-1.pb",
            'tempocnn': f"{model_path}/deepsquare-k16-3.pb",
            
            # Classifiers
            'danceability': f"{model_path}/danceability-discogs-effnet-1.pb",
            'mood_happy': f"{model_path}/mood_happy-discogs-effnet-1.pb",
            'mood_sad': f"{model_path}/mood_sad-discogs-effnet-1.pb",
            'mood_party': f"{model_path}/mood_party-discogs-effnet-1.pb",
            'mood_relaxed': f"{model_path}/mood_relaxed-discogs-effnet-1.pb",
            'mood_acoustic': f"{model_path}/mood_acoustic-discogs-effnet-1.pb",
            'voice_instrumental': f"{model_path}/voice_instrumental-discogs-effnet-1.pb",
            'genre_discogs400': f"{model_path}/genre_discogs400-discogs-effnet-1.pb",
            'timbre': f"{model_path}/timbre-discogs-effnet-1.pb",
            'deam': f"{model_path}/deam-msd-musicnn-2.pb",
            'engagement': f"{model_path}/engagement_regression-discogs-effnet-1.pb",
            'approachability': f"{model_path}/approachability_regression-discogs-effnet-1.pb",
            'mtg_jamendo_moodtheme': f"{model_path}/mtg_jamendo_moodtheme-discogs-effnet-1.pb",
        }
        
        # Cache for loaded models (avoid re-loading heavy TF graphs repeatedly)
        self._loaded_models = {}
        
        # Load models lazily - only when needed
        self.logger.info("Feature extractor initialized. Models will be loaded on first use.")
        
        # Load metadata for interpreting results
        self.load_model_metadata()
        
        # Suppress Essentia warnings
        self._setup_warning_suppression()
        
    def _setup_warning_suppression(self):
        """Suppress Essentia network warnings."""
        # Suppress the specific "No network created" warnings
        warnings.filterwarnings('ignore', message='.*No network created.*')
        warnings.filterwarnings('ignore', message='.*last created network has been deleted.*')
        
        # Also suppress TensorFlow warnings that might appear
        warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
        
    def cleanup_models(self):
        """Clean up loaded models to prevent network conflicts."""
        if hasattr(self, '_loaded_models'):
            for model_name, model in self._loaded_models.items():
                try:
                    # Try to clean up the model if it has a cleanup method
                    if hasattr(model, 'cleanup'):
                        model.cleanup()
                    elif hasattr(model, 'reset'):
                        model.reset()
                except Exception as e:
                    self.logger.debug(f"Could not cleanup model {model_name}: {e}")
            
            # Clear the loaded models cache
            self._loaded_models.clear()
            self.logger.info("Models cleaned up successfully")
        
    def load_model_metadata(self):
        """Load metadata files for interpreting model outputs."""
        self.metadata = {}
        try:
            # Load genre labels
            with open(f"{self.model_path}/genre_discogs400-discogs-effnet-1.json", 'r') as f:
                self.metadata['genre_discogs400'] = json.load(f)
            
            # Load mood theme labels
            with open(f"{self.model_path}/mtg_jamendo_moodtheme-discogs-effnet-1.json", 'r') as f:
                self.metadata['mtg_jamendo_moodtheme'] = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load model metadata files: {e}")
    
    def _load_model(self, model_name, timeout_seconds=30):
        """Load a model with timeout protection."""
        if model_name in self._loaded_models:
            return self._loaded_models[model_name]
        
        model_path = self.model_paths.get(model_name)
        if not model_path or not os.path.exists(model_path):
            self.logger.warning(f"Model file not found: {model_path}")
            return None
        
        try:
            self.logger.info(f"Loading model: {model_name}")
            with timeout(timeout_seconds):
                if model_name == 'discogs_effnet':
                    model = TensorflowPredictEffnetDiscogs(graphFilename=model_path, output="PartitionedCall:1")
                elif model_name == 'msd_musicnn':
                    model = TensorflowPredictMusiCNN(graphFilename=model_path, output="model/dense/BiasAdd")
                elif model_name == 'tempocnn':
                    model = TempoCNN(graphFilename=model_path)
                elif model_name == 'deam' or model_name == 'approachability' or model_name == 'engagement':
                    model = TensorflowPredict2D(graphFilename=model_path, output="model/Identity")
                elif model_name == 'genre_discogs400':
                    model = TensorflowPredict2D(graphFilename=model_path, output="PartitionedCall:0", input="serving_default_model_Placeholder")
                elif model_name == 'mtg_jamendo_moodtheme':
                    model = TensorflowPredict2D(graphFilename=model_path, output="model/Sigmoid")
                else:
                    # Default for most classifiers
                    model = TensorflowPredict2D(graphFilename=model_path, output="model/Softmax")
                
                self._loaded_models[model_name] = model
                self.logger.info(f"Successfully loaded model: {model_name}")
                return model
                
        except TimeoutError:
            self.logger.error(f"Timeout loading model: {model_name}")
            return None
        except Exception as e:
            self.logger.error(f"Error loading model {model_name}: {e}")
            return None
    
    def extract_features(self, audio_file, cleanup_after=True):
        """
        Extract comprehensive musical features from an audio file.
        
        Args:
            audio_file (str): Path to audio file (should be 16kHz WAV)
            cleanup_after (bool): Whether to cleanup models after extraction
            
        Returns:
            dict: Dictionary containing all extracted features
        """
        print(f"Extracting features from: {audio_file}")
        
        try:
            # Load audio at 16kHz (required for most models)
            try:
                audio = MonoLoader(filename=audio_file, sampleRate=16000, resampleQuality=4)()
                print(f"Audio loaded successfully. Duration: {len(audio)/16000:.2f} seconds")
            except Exception as e:
                print(f"Error loading audio file: {e}")
                return None
                
            features = {}
            
            # 1. Tempo Analysis (simplified approach)
            features.update(self._extract_tempo_simple(audio))
            
            # 2. Key/Scale detection
            features.update(self._extract_key_scale(audio))
            
            # 3. Embedding-based features (using Discogs-EffNet)
            effnet_embeddings = self._get_discogs_effnet_embeddings(audio)
            if effnet_embeddings is not None:
                features.update(self._extract_effnet_based_features(effnet_embeddings))
            
            # 4. Arousal/Valence (using MusiCNN embeddings)
            musicnn_embeddings = self._get_musicnn_embeddings(audio)
            if musicnn_embeddings is not None:
                features.update(self._extract_arousal_valence(musicnn_embeddings))
            
            return features
            
        finally:
            # Clean up models after extraction to prevent network conflicts
            if cleanup_after:
                self.cleanup_models()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup models."""
        self.cleanup_models()
    
    def _extract_tempo_simple(self, audio):
        """Extract tempo using a simple approach with timeout protection."""
        start_time = time.time()
        
        # Try TempoCNN first with timeout
        try:
            model = self._load_model('tempocnn', timeout_seconds=20)
            if model is not None:
                print("Extracting tempo using TempoCNN...")
                with timeout(15):  # 15 second timeout for inference
                    # Resample audio for TempoCNN
                    resampler = Resample(inputSampleRate=16000, outputSampleRate=11025)
                    audio_resampled = resampler(audio)
                    tempo_result = model(audio_resampled)
                    
                    # Parse result
                    if isinstance(tempo_result, (tuple, list)) and len(tempo_result) > 0:
                        global_tempo = float(tempo_result[0])
                    else:
                        global_tempo = float(tempo_result)
                    
                    return {
                        'tempo': {
                            'global_tempo': global_tempo,
                            'method': 'tempocnn',
                            'elapsed': time.time() - start_time
                        }
                    }
        except (TimeoutError, Exception) as e:
            print(f"TempoCNN failed: {e}")
        
        # Fallback to RhythmExtractor2013
        try:
            print("Using RhythmExtractor2013 fallback...")
            resampler = Resample(inputSampleRate=16000, outputSampleRate=44100)
            audio_44k = resampler(audio)
            
            rhythm = RhythmExtractor2013()
            rhythm_res = rhythm(audio_44k)
            
            bpm = None
            if isinstance(rhythm_res, (list, tuple)) and len(rhythm_res) > 0:
                bpm = float(rhythm_res[0])
            
            return {
                'tempo': {
                    'global_tempo': bpm,
                    'method': 'rhythm_extractor_2013',
                    'elapsed': time.time() - start_time
                }
            }
        except Exception as e:
            print(f"RhythmExtractor2013 failed: {e}")
            return {'tempo': None}
    
    def _extract_key_scale(self, audio):
        """Extract key and scale information."""
        try:
            print("Extracting key/scale...")
            key_extractor = KeyExtractor()
            key, scale, strength = key_extractor(audio)
            return {
                'key_scale': {
                    'key': str(key),
                    'scale': str(scale),
                    'strength': float(strength)
                }
            }
        except Exception as e:
            print(f"Key extraction failed: {e}")
            return {}
        
    def _get_discogs_effnet_embeddings(self, audio):
        """Get Discogs-EffNet embeddings."""
        try:
            print("Extracting Discogs-EffNet embeddings...")
            model = self._load_model('discogs_effnet', timeout_seconds=30)
            if model is None:
                return None
            
            with timeout(20):  # 20 second timeout for inference
                embeddings = model(audio)
                print("Discogs-EffNet embeddings extracted successfully")
                return embeddings
        except (TimeoutError, Exception) as e:
            print(f"Discogs-EffNet embedding extraction failed: {e}")
            return None
    
    def _get_musicnn_embeddings(self, audio):
        """Get MusiCNN embeddings."""
        try:
            print("Extracting MusiCNN embeddings...")
            model = self._load_model('msd_musicnn', timeout_seconds=30)
            if model is None:
                return None
            
            with timeout(20):  # 20 second timeout for inference
                embeddings = model(audio)
                print("MusiCNN embeddings extracted successfully")
                return embeddings
        except (TimeoutError, Exception) as e:
            print(f"MusiCNN embedding extraction failed: {e}")
            return None

    
    def _extract_effnet_based_features(self, embeddings):
        """Extract features using Discogs-EffNet embeddings."""
        if embeddings is None:
            return {}
        
        features = {}
        print("Extracting EffNet-based features...")
        start_time = time.time()
        
        # Classifier configurations
        classifier_configs = [
            ('danceability', 'danceability', ['danceable', 'not_danceable']),
            ('mood_happy', 'happiness', ['happy', 'not_happy']),
            ('voice_instrumental', 'voice_instrumental', ['instrumental', 'voice']),
            ('mood_acoustic', 'acoustic_electronic', ['acoustic', 'non_acoustic']),
            ('timbre', 'timbre', ['bright', 'dark']),
            ('mood_sad', 'sad_mood', ['sad', 'not_sad']),
            ('mood_party', 'party_mood', ['party', 'not_party']),
            ('mood_relaxed', 'relaxed_mood', ['relaxed', 'not_relaxed']),
            ('engagement', 'engagement', ['low_engagement', 'high_engagement']),
            ('approachability', 'approachability', ['low_approachability', 'high_approachability'])
        ]

        for key, feature_name, labels in classifier_configs:
            try:
                print(f"- Computing {feature_name} ({key})...")
                model = self._load_model(key, timeout_seconds=20)
                if model is None:
                    print(f"  Model not available for {key}")
                    continue

                with timeout(15):  # 15 second timeout for inference
                    preds = model(embeddings)
                    
                    # Normalize prediction shapes: take mean across time axis if present
                    try:
                        arr = np.array(preds)
                        if arr.ndim == 2:
                            vec = np.mean(arr, axis=0)
                        elif arr.ndim == 1:
                            vec = arr
                        else:
                            vec = arr.flatten()
                    except Exception:
                        vec = np.array(preds).flatten()

                    p0 = float(vec[0]) if len(vec) > 0 else 0.0
                    p1 = float(vec[1]) if len(vec) > 1 else 1.0 - p0
                    
                    features[feature_name] = {
                        f'{labels[0]}_probability': p0,
                        f'{labels[1]}_probability': p1,
                        f'is_{labels[0]}': bool(p0 > 0.5)
                    }
                    print(f"  {feature_name} done")
                    
            except (TimeoutError, Exception) as e:
                print(f"Error processing classifier {key}: {e}")

        # Handle multi-class models separately
        features.update(self._extract_multiclass_features(embeddings))
        
        print(f"All EffNet-based features extracted in {time.time() - start_time:.2f} seconds")
        return features
    
    def _extract_multiclass_features(self, embeddings):
        """Extract features from multi-class models (genre and mood theme)."""
        if embeddings is None:
            return {}
        
        features = {}
        print("Extracting multi-class features...")
        
        # Multi-class model configurations
        multiclass_configs = [
            ('genre_discogs400', 'genre', 4),  # Top 4 genres
            ('mtg_jamendo_moodtheme', 'mood_theme', 3)  # Top 3 mood themes
        ]
        
        for model_key, feature_name, top_k in multiclass_configs:
            try:
                print(f"- Computing {feature_name} ({model_key})...")
                model = self._load_model(model_key, timeout_seconds=20)
                if model is None:
                    print(f"  Model not available for {model_key}")
                    continue

                with timeout(15):  # 15 second timeout for inference
                    preds = model(embeddings)
                    
                    # Normalize prediction shapes: take mean across time axis if present
                    try:
                        arr = np.array(preds)
                        if arr.ndim == 2:
                            vec = np.mean(arr, axis=0)
                        elif arr.ndim == 1:
                            vec = arr
                        else:
                            vec = arr.flatten()
                    except Exception:
                        vec = np.array(preds).flatten()

                    # Get top-k predictions
                    top_indices = np.argsort(vec)[-top_k:][::-1]  # Top k, descending order
                    top_probs = vec[top_indices]
                    
                    # Create feature dictionary with top predictions
                    feature_dict = {}
                    
                    # Get class names from metadata if available
                    class_names = None
                    if model_key in self.metadata and 'classes' in self.metadata[model_key]:
                        class_names = self.metadata[model_key]['classes']
                    
                    for i, (idx, prob) in enumerate(zip(top_indices, top_probs)):
                        class_name = class_names[int(idx)] if class_names and int(idx) < len(class_names) else f"class_{int(idx)}"
                        feature_dict[f'top_{i+1}_class'] = class_name
                        feature_dict[f'top_{i+1}_class_id'] = int(idx)
                        feature_dict[f'top_{i+1}_probability'] = float(prob)
                    
                    # Add the most likely class
                    most_likely_name = class_names[int(top_indices[0])] if class_names and int(top_indices[0]) < len(class_names) else f"class_{int(top_indices[0])}"
                    feature_dict['most_likely_class'] = most_likely_name
                    feature_dict['most_likely_class_id'] = int(top_indices[0])
                    feature_dict['most_likely_probability'] = float(top_probs[0])
                    
                    features[feature_name] = feature_dict
                    print(f"  {feature_name} done - top class: {top_indices[0]} (prob: {top_probs[0]:.3f})")
                    
            except (TimeoutError, Exception) as e:
                print(f"Error processing multi-class model {model_key}: {e}")

        return features
    
    def _extract_arousal_valence(self, embeddings):
        """Extract arousal and valence using MusiCNN embeddings."""
        if embeddings is None:
            return {}
        
        try:
            print("Extracting arousal/valence...")
            model = self._load_model('deam', timeout_seconds=20)
            if model is None:
                return {}
            
            with timeout(15):  # 15 second timeout for inference
                predictions = model(embeddings)
                
                return {
                    'arousal_valence': {
                        'valence': float(predictions[0][0]),  # Range [1, 9]
                        'arousal': float(predictions[0][1]),  # Range [1, 9]
                        'valence_normalized': float((predictions[0][0] - 1) / 8),  # Normalize to [0, 1]
                        'arousal_normalized': float((predictions[0][1] - 1) / 8)   # Normalize to [0, 1]
                    }
                }
        except (TimeoutError, Exception) as e:
            print(f"Error extracting arousal/valence: {e}")
            return {}
    