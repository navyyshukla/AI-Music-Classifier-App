import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from tqdm import tqdm
import warnings

# Suppress annoying librosa warnings to keep the output clean
warnings.filterwarnings('ignore', category=UserWarning)

class SpectrogramGenerator:
    """
    An advanced class to process the GTZAN audio dataset and generate spectrograms
    with data augmentation to improve model accuracy.
    """
    def __init__(self, input_dir, output_dir, n_mels=128, fmax=8000, duration=30):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.n_mels = n_mels
        self.fmax = fmax
        self.duration = duration
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}")

    # --- AUGMENTATION ALGORITHMS ---

    def _add_noise(self, y, noise_factor=0.005):
        """Adds random gaussian noise to an audio signal."""
        noise = np.random.randn(len(y))
        augmented_y = y + noise_factor * noise
        return augmented_y.astype(type(y[0]))

    def _time_stretch(self, y, stretch_rate=0.8):
        """Stretches the time of an audio signal without changing pitch."""
        return librosa.effects.time_stretch(y=y, rate=stretch_rate)

    def _pitch_shift(self, y, sr, n_steps=4):
        """Shifts the pitch of an audio signal."""
        return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)

    # --- CORE SPECTROGRAM CREATION ---

    def create_spectrogram_image(self, y, sr, output_path):
        """Creates and saves a Mel Spectrogram image from an audio signal."""
        try:
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels, fmax=self.fmax)
            S_dB = librosa.power_to_db(S, ref=np.max)
            
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=self.fmax)
            
            # This part is crucial: remove all padding, axes, and labels
            # to create a clean image for the CNN.
            plt.gca().set_axis_off()
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close()
        except Exception as e:
            print(f"  [Error] Could not create spectrogram for {output_path}: {e}")

    # --- MAIN PROCESSING PIPELINE ---
    
    def process(self):
        """
        Processes the entire dataset, creating original and augmented spectrograms.
        This function is scalable and will work even if you add more genre folders later.
        """
        genres = sorted([g for g in os.listdir(self.input_dir) if os.path.isdir(os.path.join(self.input_dir, g))])
        
        for genre in genres:
            input_genre_path = os.path.join(self.input_dir, genre)
            output_genre_path = os.path.join(self.output_dir, genre)
            if not os.path.exists(output_genre_path):
                os.makedirs(output_genre_path)

            audio_files = sorted([f for f in os.listdir(input_genre_path) if f.endswith('.wav')])
            
            print(f"\nProcessing genre: {genre} ({len(audio_files)} original files)")
            
            # The tqdm wrapper creates a beautiful progress bar
            for filename in tqdm(audio_files, desc=f"  -> {genre}", ncols=100):
                audio_path = os.path.join(input_genre_path, filename)
                base_filename = os.path.splitext(filename)[0]
                
                try:
                    # 1. Load original audio
                    y, sr = librosa.load(audio_path, duration=self.duration)
                    if len(y) < self.duration * sr:
                        tqdm.write(f"  [Skipping] {filename} is too short.")
                        continue

                    # 2. Create and save the ORIGINAL spectrogram
                    original_output_path = os.path.join(output_genre_path, f"{base_filename}_original.png")
                    self.create_spectrogram_image(y, sr, original_output_path)
                    
                    # 3. Create and save AUGMENTED spectrograms
                    # Noise
                    y_noise = self._add_noise(y)
                    noise_output_path = os.path.join(output_genre_path, f"{base_filename}_noise.png")
                    self.create_spectrogram_image(y_noise, sr, noise_output_path)
                    
                    # Time Stretch
                    y_stretch = self._time_stretch(y)
                    stretch_output_path = os.path.join(output_genre_path, f"{base_filename}_stretch.png")
                    self.create_spectrogram_image(y_stretch, sr, stretch_output_path)

                    # Pitch Shift
                    y_pitch = self._pitch_shift(y, sr)
                    pitch_output_path = os.path.join(output_genre_path, f"{base_filename}_pitch.png")
                    self.create_spectrogram_image(y_pitch, sr, pitch_output_path)

                except Exception as e:
                    tqdm.write(f"  [Error] Failed processing file {filename}: {e}")

        print("\n--- Advanced feature extraction complete! ---")
        print("Dataset has been augmented and converted to spectrogram images.")

if __name__ == "__main__":
    INPUT_DATASET_PATH = 'Data/genres_original'
    # We save to a new folder to keep things clean
    OUTPUT_SPECTROGRAMS_PATH = 'spectrograms_augmented'
    
    generator = SpectrogramGenerator(INPUT_DATASET_PATH, OUTPUT_SPECTROGRAMS_PATH)
    generator.process()
