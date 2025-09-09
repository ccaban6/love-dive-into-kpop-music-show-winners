import os
import json
import sqlite3
import pandas as pd
from datetime import datetime
import re
from pathlib import Path
import librosa
import warnings

# Import our custom modules
from downloader import download_song, verify_audio_specs
from feature_extractor import EssentiaFeatureExtractor

class AudioProcessingPipeline:
    def __init__(self, 
                 db_path="data/sql/clean.db",
                 output_dir="data/audio/raw",
                 processed_dir="data/processed",
                 cache_file="data/processed/processing_cache.json",
                 project_root=None):
        """
        Initialize the audio processing pipeline.
        
        Args:
            db_path (str): Path to SQLite database
            output_dir (str): Directory for downloaded audio files
            processed_dir (str): Directory for processed metadata
            cache_file (str): Path to cache file for tracking processed files
            project_root (str): Root directory of the project (auto-detected if None)
        """
        # Auto-detect project root if not provided
        if project_root is None:
            # Find the project root by looking for common markers
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)  # Go up one level from scripts/
        
        # Convert relative paths to absolute paths
        self.project_root = project_root
        self.db_path = os.path.join(project_root, db_path) if not os.path.isabs(db_path) else db_path
        self.output_dir = os.path.join(project_root, output_dir) if not os.path.isabs(output_dir) else output_dir
        self.processed_dir = os.path.join(project_root, processed_dir) if not os.path.isabs(processed_dir) else processed_dir
        self.cache_file = os.path.join(project_root, cache_file) if not os.path.isabs(cache_file) else cache_file
        
        # Create directories if they don't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Initialize cache
        self.cache = self._load_cache()
        
        # Initialize feature extractor
        self.feature_extractor = EssentiaFeatureExtractor(model_path="../essentia_models")
        
        print(f"Audio Processing Pipeline initialized")
        print(f"Database: {db_path}")
        print(f"Output directory: {output_dir}")
        print(f"Processed directory: {processed_dir}")
        print(f"Cache file: {cache_file}")
    
    def _load_cache(self):
        """Load the processing cache from JSON file."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load cache file: {e}")
                return {}
        return {}
    
    def _save_cache(self):
        """Save the processing cache to JSON file."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save cache file: {e}")
    
    def _extract_metadata_from_filename(self, filename):
        """
        Extract artist and song title from audio filename.
        Expected format: "Artist - Song Title.ext"
        """
        # Remove file extension
        name_without_ext = os.path.splitext(filename)[0]
        
        # Split by " - " (space, dash, space)
        parts = name_without_ext.split(" - ", 1)
        
        if len(parts) == 2:
            artist = parts[0].strip()
            song_title = parts[1].strip()
        else:
            # Fallback: try other common separators
            separators = [" - ", " – ", " — ", " -", "- ", " –", "– ", " —", "— "]
            for sep in separators:
                if sep in name_without_ext:
                    parts = name_without_ext.split(sep, 1)
                    if len(parts) == 2:
                        artist = parts[0].strip()
                        song_title = parts[1].strip()
                        break
            else:
                # If no separator found, assume the whole name is the song title
                artist = "Unknown"
                song_title = name_without_ext.strip()
        
        return {
            'artist': artist,
            'song_title': song_title,
            'filename': filename
        }
    
    def _get_audio_duration(self, audio_file_path):
        """Get duration of audio file in seconds."""
        try:
            # Use librosa to get duration
            duration = librosa.get_duration(filename=audio_file_path)
            return float(duration)
        except Exception as e:
            print(f"Warning: Could not get duration for {audio_file_path}: {e}")
            return None
    
    def _is_file_processed(self, filename):
        """Check if a file has already been processed."""
        return filename in self.cache
    
    def _is_artist_song_processed(self, artist, song_title):
        """Check if an artist-song combination has already been processed."""
        for filename, cache_data in self.cache.items():
            metadata = cache_data.get('metadata', {})
            if (metadata.get('artist', '').lower() == artist.lower() and 
                metadata.get('song_title', '').lower() == song_title.lower()):
                return True, filename
        return False, None
    
    def _mark_file_processed(self, filename, metadata):
        """Mark a file as processed in the cache."""
        self.cache[filename] = {
            'processed_at': datetime.now().isoformat(),
            'metadata': metadata
        }
        self._save_cache()
    
    def process_single_audio_file(self, audio_file_path, force_reprocess=False, download_metadata=None):
        """
        Process a single audio file: extract features and metadata.
        
        Args:
            audio_file_path (str): Path to audio file
            force_reprocess (bool): Force reprocessing even if already processed
            download_metadata (dict): Optional metadata from download process (includes URL)
            
        Returns:
            dict: Complete metadata dictionary with features
        """
        filename = os.path.basename(audio_file_path)
        
        # Check if already processed
        if not force_reprocess and self._is_file_processed(filename):
            print(f"File {filename} already processed, skipping...")
            return self.cache[filename]['metadata']
        
        print(f"Processing audio file: {filename}")
        
        # Extract basic metadata from filename
        basic_metadata = self._extract_metadata_from_filename(filename)
        
        # Get audio duration
        duration = self._get_audio_duration(audio_file_path)
        basic_metadata['duration_seconds'] = duration
        
        # Extract audio features
        try:
            print("Extracting audio features...")
            features = self.feature_extractor.extract_features(audio_file_path)
            
            if features is None:
                print(f"Warning: Feature extraction failed for {filename}")
                features = {}
            
            # Combine basic metadata with features
            complete_metadata = {
                **basic_metadata,
                'audio_features': features,
                'processing_timestamp': datetime.now().isoformat(),
                'file_path': audio_file_path
            }
            
            # Add download metadata if available (includes URL for verification)
            if download_metadata:
                complete_metadata.update({
                    'download_url': download_metadata.get('url'),
                    'download_title': download_metadata.get('title'),
                    'download_uploader': download_metadata.get('uploader'),
                    'download_duration_sec': download_metadata.get('duration_sec'),
                    'downloaded_at': download_metadata.get('downloaded_at')
                })
            
            # Mark as processed
            self._mark_file_processed(filename, complete_metadata)
            
            print(f"Successfully processed {filename}")
            return complete_metadata
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            return None
    
    def download_and_process_song(self, song_name, artist, min_duration=60, max_duration=600, force_reprocess=False):
        """
        Download a song and process it through the pipeline.
        
        Args:
            song_name (str): Name of the song
            artist (str): Name of the artist
            min_duration (int): Minimum acceptable duration in seconds
            max_duration (int): Maximum acceptable duration in seconds
            force_reprocess (bool): Force reprocessing even if already processed
            
        Returns:
            dict: Complete metadata dictionary or None if failed
        """
        # Check if artist-song combination already processed (case-insensitive)
        if not force_reprocess:
            is_processed, existing_filename = self._is_artist_song_processed(artist, song_name)
            if is_processed:
                print(f"Artist-song combination '{artist} - {song_name}' already processed (file: {existing_filename}), skipping...")
                return self.cache[existing_filename]['metadata']
        
        # Check if file already exists
        expected_filename = f"{artist} - {song_name}.wav"
        expected_path = os.path.join(self.output_dir, expected_filename)
        
        if os.path.exists(expected_path) and not self._is_file_processed(expected_filename):
            print(f"File {expected_filename} exists but not processed, processing now...")
            return self.process_single_audio_file(expected_path, force_reprocess)
        elif os.path.exists(expected_path) and self._is_file_processed(expected_filename) and not force_reprocess:
            print(f"File {expected_filename} already exists and processed, skipping...")
            return self.cache[expected_filename]['metadata']
        
        # Download the song
        print(f"Downloading {artist} - {song_name}...")
        # Temporarily change the output directory for the downloader
        import downloader
        original_output_dir = downloader.OUTPUT_DIR
        downloader.OUTPUT_DIR = self.output_dir
        
        try:
            download_metadata = download_song(song_name, artist, min_duration, max_duration)
        finally:
            # Restore original output directory
            downloader.OUTPUT_DIR = original_output_dir
        
        if download_metadata is None:
            print(f"Failed to download {artist} - {song_name}")
            return None
        
        # Process the downloaded file
        downloaded_path = os.path.join(self.output_dir, expected_filename)
        if os.path.exists(downloaded_path):
            # Pass download metadata (including URL) to the processing function
            return self.process_single_audio_file(downloaded_path, download_metadata=download_metadata)
        else:
            print(f"Downloaded file not found at expected path: {downloaded_path}")
            return None
    
    def process_songs_from_database(self, table_name="songs", limit=None, force_reprocess=False, 
                                   song_column="song", artist_column="artist"):
        """
        Process songs from SQLite database.
        
        Args:
            table_name (str): Name of the table containing song data
            limit (int): Maximum number of songs to process (None for all)
            force_reprocess (bool): Force reprocessing of already processed files
            song_column (str): Name of the column containing song titles
            artist_column (str): Name of the column containing artist names
            
        Returns:
            list: List of processed metadata dictionaries
        """
        try:
            # Connect to database
            conn = sqlite3.connect(self.db_path)
            
            # Query unique song-artist combinations from database
            # This is much more efficient than selecting all rows and deduplicating in Python
            query = f"SELECT DISTINCT {song_column}, {artist_column} FROM {table_name}"
            if limit:
                query += f" LIMIT {limit}"
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            print(f"Found {len(df)} unique song-artist combinations in database")
            
            processed_songs = []
            skipped_already_processed = 0
            
            for idx, row in df.iterrows():
                print(f"\nProcessing song {idx + 1}/{len(df)}")
                
                # Extract song information from database row
                song_name = row.get(song_column, '').strip()
                artist = row.get(artist_column, '').strip()
                
                if not song_name or not artist:
                    print(f"Skipping row {idx}: missing song name or artist")
                    continue
                
                # Download and process
                metadata = self.download_and_process_song(song_name, artist, force_reprocess=force_reprocess)
                
                if metadata:
                    processed_songs.append(metadata)
                    print(f"Successfully processed: {artist} - {song_name}")
                else:
                    print(f"Failed to process: {artist} - {song_name}")
            
            print(f"\nProcessing Summary:")
            print(f"  Unique combinations in database: {len(df)}")
            print(f"  Successfully processed: {len(processed_songs)}")
            print(f"  Failed to process: {len(df) - len(processed_songs)}")
            
            return processed_songs
            
        except Exception as e:
            print(f"Error processing songs from database: {e}")
            return []
    
    def save_processed_metadata(self, metadata_list, output_filename=None):
        """
        Save processed metadata to JSON file.
        
        Args:
            metadata_list (list): List of metadata dictionaries
            output_filename (str): Output filename (auto-generated if None)
        """
        if not output_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"processed_audio_metadata_{timestamp}.json"
        
        output_path = os.path.join(self.processed_dir, output_filename)
        
        try:
            with open(output_path, 'w') as f:
                json.dump(metadata_list, f, indent=2)
            
            print(f"Saved processed metadata to: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error saving metadata: {e}")
            return None
    
    def get_processing_summary(self):
        """Get a summary of processed files."""
        total_processed = len(self.cache)
        print(f"\nProcessing Summary:")
        print(f"Total files processed: {total_processed}")
        
        if total_processed > 0:
            # Show recent files
            recent_files = list(self.cache.keys())[-5:]
            print(f"Recent files processed:")
            for filename in recent_files:
                processed_at = self.cache[filename].get('processed_at', 'Unknown')
                print(f"  - {filename} (processed: {processed_at})")
        
        return total_processed
    
    def get_download_urls(self, show_all=False, limit=10):
        """
        Get URLs for downloaded songs for verification purposes.
        
        Args:
            show_all (bool): Show all processed files (default: False, shows recent)
            limit (int): Maximum number of URLs to show (ignored if show_all=True)
            
        Returns:
            list: List of dictionaries with song info and URLs
        """
        urls_info = []
        
        for filename, cache_data in self.cache.items():
            metadata = cache_data.get('metadata', {})
            download_url = metadata.get('download_url')
            
            if download_url:  # Only include files that have URLs
                urls_info.append({
                    'filename': filename,
                    'artist': metadata.get('artist', 'Unknown'),
                    'song_title': metadata.get('song_title', 'Unknown'),
                    'url': download_url,
                    'download_title': metadata.get('download_title', 'Unknown'),
                    'download_uploader': metadata.get('download_uploader', 'Unknown'),
                    'duration_sec': metadata.get('download_duration_sec', 'Unknown'),
                    'processed_at': cache_data.get('processed_at', 'Unknown')
                })
        
        # Sort by processing time (most recent first)
        urls_info.sort(key=lambda x: x['processed_at'], reverse=True)
        
        if not show_all and len(urls_info) > limit:
            urls_info = urls_info[:limit]
        
        print(f"\nDownload URLs for Verification:")
        print(f"{'='*80}")
        
        if not urls_info:
            print("No URLs found in processed files.")
            return []
        
        for i, info in enumerate(urls_info, 1):
            print(f"\n{i}. {info['artist']} - {info['song_title']}")
            print(f"   File: {info['filename']}")
            print(f"   URL: {info['url']}")
            print(f"   YouTube Title: {info['download_title']}")
            print(f"   Uploader: {info['download_uploader']}")
            print(f"   Duration: {info['duration_sec']} seconds")
            print(f"   Processed: {info['processed_at']}")
        
        if not show_all and len(self.cache) > limit:
            print(f"\n... and {len(self.cache) - limit} more files. Use show_all=True to see all.")
        
        return urls_info
    
    def get_url_for_song(self, artist, song_title):
        """
        Get the download URL for a specific song.
        
        Args:
            artist (str): Artist name
            song_title (str): Song title
            
        Returns:
            str: URL if found, None otherwise
        """
        for filename, cache_data in self.cache.items():
            metadata = cache_data.get('metadata', {})
            if (metadata.get('artist', '').lower() == artist.lower() and 
                metadata.get('song_title', '').lower() == song_title.lower()):
                return metadata.get('download_url')
        return None
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'feature_extractor'):
            self.feature_extractor.cleanup_models()
        print("Pipeline cleanup completed")


def main():
    """Example usage of the audio processing pipeline."""
    
    # Initialize pipeline
    pipeline = AudioProcessingPipeline(
        db_path="data/sql/clean.db",
        output_dir="data/audio/raw",
        processed_dir="data/processed"
    )
    
    try:
        # Example 1: Process a single song
        print("=== Processing Single Song ===")
        metadata = pipeline.download_and_process_song("World", "SEVENTEEN")
        if metadata:
            print(f"Processed: {metadata['artist']} - {metadata['song_title']}")
            print(f"Duration: {metadata['duration_seconds']:.2f} seconds")
        
        # Example 2: Process songs from database
        print("\n=== Processing Songs from Database ===")
        processed_songs = pipeline.process_songs_from_database(limit=5)
        
        # Example 3: Save processed metadata
        if processed_songs:
            output_file = pipeline.save_processed_metadata(processed_songs)
            print(f"Saved metadata to: {output_file}")
        
        # Example 4: Show processing summary
        pipeline.get_processing_summary()
        
        # Example 5: Show download URLs for verification
        print("\n=== Download URLs for Verification ===")
        pipeline.get_download_urls(limit=5)
        
    finally:
        # Cleanup
        pipeline.cleanup()


if __name__ == "__main__":
    main()
