# Utility functions module
import os
import pickle
import hashlib
import json
import shutil
import subprocess
import re
import time
from collections import Counter

# Constants
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Default data (will be imported from config)
DEFAULT_DATA = {
    'chosen_process_method': 'vr',
    'vr_model': 'Choose Model',
    'mdx_net_model': 'Choose Model',
    'demucs_model': 'Choose Model',
    'export_path': os.path.join(BASE_PATH, 'output'),
    'input_paths': (),
    'lastDir': '',
    'is_gpu_conversion': True,
    'is_primary_stem_only': False,
    'is_secondary_stem_only': False,
    'is_testing_audio': False,
    'is_add_model_name': False,
    'is_accept_any_input': False,
    'is_task_complete': True,
    'is_normalization': False,
    'is_use_opencl': False,
    'is_wav_ensemble': False,
    'is_create_model_folder': False,
    'mp3_bit_set': '320',
    'save_format': 'WAV',
    'wav_type_set': 'PCM_16',
    'device_set': 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu',
    'user_code': '',
    'help_hints_var': True,
    'model_sample_mode': False,
    'model_sample_mode_duration': '30',
    'set_vocal_splitter': 'No Model',
    'is_set_vocal_splitter': False,
    'is_save_inst_set_vocal_splitter': False,
    # VR Architecture Vars
    'aggression_setting': '10',
    'window_size': '512',
    'batch_size': 'def',
    'crop_size': '1024',
    'is_tta': False,
    'is_output_image': False,
    'is_post_process': False,
    'is_high_end_process': False,
    'post_process_threshold': '0.5',
    'vr_is_secondary_model_activate': False,
    'vr_voc_inst_secondary_model': 'No Model',
    'vr_other_secondary_model': 'No Model',
    'vr_bass_secondary_model': 'No Model',
    'vr_drums_secondary_model': 'No Model',
    'vr_voc_inst_secondary_model_scale': '0.5',
    'vr_other_secondary_model_scale': '0.5',
    'vr_bass_secondary_model_scale': '0.5',
    'vr_drums_secondary_model_scale': '0.5',
    # MDX-Net Vars
    'chunks': '0',
    'margin': '10',
    'compensate': 'auto',
    'denoise_option': 'none',
    'phase_option': 'none',
    'phase_shifts': '0',
    'is_save_align': False,
    'is_match_silence': False,
    'is_spec_match': False,
    'is_match_frequency_pitch': False,
    'is_mdx_c_seg_def': False,
    'is_invert_spec': False,
    'is_deverb_vocals': False,
    'deverb_vocal_opt': 'All Vocals',
    'voc_split_save_opt': 'All Vocals',
    'is_mixer_mode': False,
    'mdx_batch_size': 'def',
    'mdx_is_secondary_model_activate': False,
    'mdx_voc_inst_secondary_model': 'No Model',
    'mdx_other_secondary_model': 'No Model',
    'mdx_bass_secondary_model': 'No Model',
    'mdx_drums_secondary_model': 'No Model',
    'mdx_voc_inst_secondary_model_scale': '0.5',
    'mdx_other_secondary_model_scale': '0.5',
    'mdx_bass_secondary_model_scale': '0.5',
    'mdx_drums_secondary_model_scale': '0.5',
    # Demucs Vars
    'segment': '10',
    'overlap': '0.25',
    'overlap_mdx': 'default',
    'overlap_mdx23': '8',
    'shifts': '1',
    'chunks_demucs': '0',
    'margin_demucs': '10',
    'is_chunk_demucs': False,
    'is_chunk_mdxnet': False,
    'is_primary_stem_only_Demucs': False,
    'is_secondary_stem_only_Demucs': False,
    'is_split_mode': False,
    'is_demucs_combine_stems': False,
    'is_mdx23_combine_stems': False,
    'demucs_is_secondary_model_activate': False,
    'demucs_voc_inst_secondary_model': 'No Model',
    'demucs_other_secondary_model': 'No Model',
    'demucs_bass_secondary_model': 'No Model',
    'demucs_drums_secondary_model': 'No Model',
    'demucs_voc_inst_secondary_model_scale': '0.5',
    'demucs_other_secondary_model_scale': '0.5',
    'demucs_bass_secondary_model_scale': '0.5',
    'demucs_drums_secondary_model_scale': '0.5',
    'demucs_pre_proc_model': 'No Model',
    'is_demucs_pre_proc_model_activate': False,
    'is_demucs_pre_proc_model_inst_mix': False,
    # Ensemble Vars
    'is_save_all_outputs_ensemble': False,
    'is_append_ensemble_name': False,
    'chosen_ensemble': 'Choose Ensemble Option',
    'ensemble_type': 'Mean',
    'ensemble_main_stem': 'Vocals/Instrumental',
    # Audio Tool Vars
    'chosen_audio_tool': 'Time Stretch',
    'choose_algorithm': 'Mean',
    'time_stretch_rate': '1.0',
    'pitch_rate': '0',
    'is_time_correction': False,
    # Shared Vars
    'semitone_shift': '0',
    'mdx_segment_size': '256',
    'denoise_option': 'none',
    'phase_option': 'none',
    'phase_shifts': '0',
    'is_save_align': False,
    'is_match_silence': False,
    'is_spec_match': False,
    'is_match_frequency_pitch': False,
    'is_mdx_c_seg_def': False,
    'is_invert_spec': False,
    'is_deverb_vocals': False,
    'deverb_vocal_opt': 'All Vocals',
    'voc_split_save_opt': 'All Vocals',
    'is_mixer_mode': False,
    'fileOneEntry': '',
    'fileOneEntry_Full': '',
    'fileTwoEntry': '',
    'fileTwoEntry_Full': '',
    'DualBatch_inputPaths': [],
    'time_window': '10',
    'intro_analysis': '0',
    'db_analysis': '0',
    'demucs_stems': 'vocals',
    'mdxnet_stems': 'vocals',
    'is_auto_update_model_params': True,
}

# Create output directory if it doesn't exist
if not os.path.isdir(DEFAULT_DATA['export_path']):
    os.makedirs(DEFAULT_DATA['export_path'])

def save_data(data):
    """
    Saves given data as a .pkl (pickle) file

    Parameters:
        data(dict):
            Dictionary containing all the necessary data to save
    """
    # Open data file, create it if it does not exist
    data_file_path = os.path.join(BASE_PATH, 'data.pkl')
    with open(data_file_path, 'wb') as data_file:
        pickle.dump(data, data_file)

def load_data() -> dict:
    """
    Loads saved pkl file and returns the stored data

    Returns(dict):
        Dictionary containing all the saved data
    """
    data_file_path = os.path.join(BASE_PATH, 'data.pkl')
    
    try:
        with open(data_file_path, 'rb') as data_file:  # Open data file
            data = pickle.load(data_file)

        return data
    except (ValueError, FileNotFoundError):
        # Data File is corrupted or not found so recreate it
        save_data(data=DEFAULT_DATA)
        return load_data()

def load_model_hash_data(dictionary):
    '''Get the model hash dictionary'''
    with open(dictionary, 'r') as d:
        return json.load(d)

def font_checker(font_file):
    """Check font file and return font name and file path"""
    chosen_font_name = None
    chosen_font_file = None
    
    try:
        if os.path.isfile(font_file):
            with open(font_file, 'r') as d:
                chosen_font = json.load(d)
                
            chosen_font_name = chosen_font["font_name"]
            if chosen_font["font_file"]:
                other_font_path = os.path.join(BASE_PATH, 'gui_data', 'fonts', 'other')
                chosen_font_file = os.path.join(other_font_path, chosen_font["font_file"])
                chosen_font_file = chosen_font_file if os.path.isfile(chosen_font_file) else None
    except Exception as e:
        print(f"Error checking font: {e}")
        
    chosen_font = chosen_font_name, chosen_font_file
    
    return chosen_font

def remove_temps(temp_dir):
    """Remove temporary files and directories"""
    if os.path.isdir(temp_dir):
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Error removing temp directory {temp_dir}: {e}")

def extract_stems(audio_file_base, export_path):
    """Extract stems from exported files"""
    filenames = [file for file in os.listdir(export_path) if file.startswith(audio_file_base)]

    pattern = r'\(([^()]+)\)(?=[^()]*\.wav)'
    stem_list = []

    for filename in filenames:
        match = re.search(pattern, filename)
        if match:
            stem_list.append(match.group(1))
            
    counter = Counter(stem_list)
    filtered_lst = [item for item in stem_list if counter[item] > 1]

    return list(set(filtered_lst))

def get_execution_time(function, name):
    """Measure execution time of a function"""
    start = time.time()
    function()
    end = time.time()
    time_difference = end - start
    print(f'{name} Execution Time: ', time_difference)
    return time_difference

def vip_downloads(password, link_type):
    """Attempts to decrypt VIP model link with given input code"""
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    import base64
    
    try:
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=link_type[0],
            iterations=390000,)

        key = base64.urlsafe_b64encode(kdf.derive(bytes(password, 'utf-8')))
        f = Fernet(key)

        return str(f.decrypt(link_type[1]), 'UTF-8')
    except Exception:
        return 'Invalid code'

def verify_audio(audio_file):
    """Verify if audio file is valid"""
    import audioread
    
    try:
        with audioread.audio_open(audio_file):
            return True
    except Exception:
        return False

def create_sample(audio_file, duration=30):
    """Create a sample of the audio file"""
    import soundfile as sf
    import librosa
    
    try:
        # Load audio
        y, sr = librosa.load(audio_file, duration=duration)
        
        # Create sample path
        sample_path = os.path.join(BASE_PATH, 'temp_sample_clips')
        if not os.path.isdir(sample_path):
            os.makedirs(sample_path)
        
        sample_file = os.path.join(sample_path, f'sample_{os.path.basename(audio_file)}')
        
        # Save sample
        sf.write(sample_file, y, sr)
        
        return sample_file
    except Exception as e:
        print(f"Error creating sample: {e}")
        return audio_file

def process_input_selections(input_paths, accept_any_input=False):
    """Process input selections and return valid audio files"""
    import natsort
    
    input_list = []
    
    # Define valid extensions
    if accept_any_input:
        ext = ('.wav', '.mp3', '.aac', '.flac', '.ogg', '.m4a', '.wma')
    else:
        ext = ('.wav', '.mp3')

    for path in input_paths:
        if os.path.isfile(path):
            if path.lower().endswith(ext):
                input_list.append(path)
        elif os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.lower().endswith(ext):
                        file_path = os.path.join(root, file)
                        if os.path.isfile(file_path):
                            input_list.append(file_path)
    
    # Sort files
    input_list = natsort.natsorted(input_list)
    
    return tuple(input_list)

def process_storage_check():
    """Verify storage requirements"""
    total, used, free = shutil.disk_usage("/") 
    
    space_details = f"Detected Total Space: {int(total/1.074e+9)} GB's\n"
    space_details += f"Detected Used Space: {int(used/1.074e+9)} GB's\n"
    space_details += f"Detected Free Space: {int(free/1.074e+9)} GB's\n"
        
    appropriate_storage = True
        
    if int(free/1.074e+9) <= 2:
        print(f"Storage error: {space_details}")
        appropriate_storage = False
    elif int(free/1.074e+9) in [3, 4, 5, 6, 7, 8]:
        print(f"Storage warning: {space_details}")
        # In CLI mode, we'll continue anyway
        appropriate_storage = True
                    
    return appropriate_storage

def clear_gpu_cache():
    """Clear GPU cache"""
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
