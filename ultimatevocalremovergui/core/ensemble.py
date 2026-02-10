# Ensemble module
import os
import sys
import time
from pathlib import Path
import shutil

# Constants
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENSEMBLE_TEMP_PATH = os.path.join(BASE_PATH, 'ensemble_temps')

# Add BASE_PATH to Python path
sys.path.insert(0, BASE_PATH)

# Imports
from lib_v5 import spec_utils

class Ensembler:
    def __init__(self, is_manual_ensemble=False, config=None):
        # Use config object instead of root variables
        self.config = config or {}
        
        self.is_save_all_outputs_ensemble = self.config.get('is_save_all_outputs_ensemble', False)
        chosen_ensemble_name = self.config.get('chosen_ensemble', 'Ensembled')
        if chosen_ensemble_name == 'Choose Ensemble Option':
            chosen_ensemble_name = 'Ensembled'
        
        # Handle ensemble name
        chosen_ensemble_name = chosen_ensemble_name.replace(" ", "_")
        
        # Get ensemble algorithm and stems
        ensemble_algorithm = self.config.get('ensemble_type', 'Mean').split("/")
        ensemble_main_stem_pair = self.config.get('ensemble_main_stem', 'Vocals/Instrumental').split("/")
        time_stamp = round(time.time())
        
        self.audio_tool = 'manual_ensemble' if is_manual_ensemble else 'ensemble'
        self.main_export_path = Path(self.config.get('export_path', os.getcwd()))
        self.chosen_ensemble = f"_{chosen_ensemble_name}" if self.config.get('is_append_ensemble_name', False) else ''
        
        # Determine ensemble folder name
        ensemble_folder_name = self.main_export_path if self.is_save_all_outputs_ensemble else ENSEMBLE_TEMP_PATH
        self.ensemble_folder_name = os.path.join(ensemble_folder_name, f'{chosen_ensemble_name}_Outputs_{time_stamp}')
        
        self.is_testing_audio = f"{time_stamp}_" if self.config.get('is_testing_audio', False) else ''
        self.primary_algorithm = ensemble_algorithm[0]
        self.secondary_algorithm = ensemble_algorithm[1] if len(ensemble_algorithm) > 1 else ensemble_algorithm[0]
        self.ensemble_primary_stem = ensemble_main_stem_pair[0]
        self.ensemble_secondary_stem = ensemble_main_stem_pair[1] if len(ensemble_main_stem_pair) > 1 else 'Instrumental'
        
        # Get audio settings
        self.is_normalization = self.config.get('is_normalization', False)
        self.is_wav_ensemble = self.config.get('is_wav_ensemble', False)
        self.wav_type_set = self.config.get('wav_type_set', '16-bit Integer')
        self.mp3_bit_set = self.config.get('mp3_bit_set', '320')
        self.save_format = self.config.get('save_format', 'wav')
        
        # Create ensemble folder if not manual ensemble
        if not is_manual_ensemble:
            if not os.path.isdir(self.ensemble_folder_name):
                os.makedirs(self.ensemble_folder_name)

    def ensemble_outputs(self, audio_file_base, export_path, stem, is_4_stem=False, is_inst_mix=False):
        """Processes the given outputs and ensembles them with the chosen algorithm"""
        
        # Determine algorithm and stem tag
        if is_4_stem:
            algorithm = self.config.get('ensemble_type', 'Mean')
            stem_tag = stem
        else:
            if is_inst_mix:
                algorithm = self.secondary_algorithm
                stem_tag = f"{self.ensemble_secondary_stem} Instrumental"
            else:
                algorithm = self.primary_algorithm if stem == 'Primary Stem' else self.secondary_algorithm
                stem_tag = self.ensemble_primary_stem if stem == 'Primary Stem' else self.ensemble_secondary_stem

        # Get files to ensemble
        stem_outputs = self.get_files_to_ensemble(folder=export_path, prefix=audio_file_base, suffix=f"_({stem_tag}).wav")
        
        # Determine output file path
        audio_file_output = f"{self.is_testing_audio}{audio_file_base}{self.chosen_ensemble}_({stem_tag})"
        stem_save_path = os.path.join(str(self.main_export_path), f'{audio_file_output}.wav')
        
        # Ensemble the outputs
        if len(stem_outputs) > 1:
            spec_utils.ensemble_inputs(
                stem_outputs, 
                algorithm, 
                self.is_normalization, 
                self.wav_type_set, 
                stem_save_path, 
                is_wave=self.is_wav_ensemble
            )
            # Save format
            self.save_format_file(stem_save_path)
        
        # Handle save all outputs option
        if self.is_save_all_outputs_ensemble:
            for i in stem_outputs:
                self.save_format_file(i)
        else:
            # Clean up temporary files
            for i in stem_outputs:
                try:
                    os.remove(i)
                except Exception as e:
                    print(f"Error removing file {i}: {e}")

    def ensemble_manual(self, audio_inputs, audio_file_base, is_bulk=False):
        """Processes the given outputs and ensembles them with the chosen algorithm"""
        
        # Get algorithm
        algorithm = self.config.get('choose_algorithm', 'Mean')
        
        # Determine output file path
        audio_file_output = f"{self.is_testing_audio}{audio_file_base}{self.chosen_ensemble}"
        stem_save_path = os.path.join(str(self.main_export_path), f'{audio_file_output}.wav')
        
        # Ensemble the inputs
        if len(audio_inputs) > 1:
            spec_utils.ensemble_inputs(
                audio_inputs, 
                algorithm, 
                self.is_normalization, 
                self.wav_type_set, 
                stem_save_path, 
                is_wave=self.is_wav_ensemble
            )
            # Save format
            self.save_format_file(stem_save_path)
        
        return stem_save_path

    def get_files_to_ensemble(self, folder, prefix, suffix):
        """Get files to ensemble based on prefix and suffix"""
        files = []
        
        # Iterate through files in the folder
        for file_name in os.listdir(folder):
            if file_name.startswith(prefix) and file_name.endswith(suffix):
                files.append(os.path.join(folder, file_name))
        
        return files

    def save_format_file(self, file_path):
        """Save file in the specified format"""
        from separate import save_format
        
        try:
            save_format(file_path, self.save_format, self.mp3_bit_set)
        except Exception as e:
            print(f"Error saving format for {file_path}: {e}")

    def combine_audio(self, input_paths, audio_file_base):
        """Combine multiple audio files"""
        # This method will be implemented if needed
        pass
