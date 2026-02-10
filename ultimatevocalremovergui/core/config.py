# Configuration management module
import os
import json
import pickle
from pathlib import Path

# Constants
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(BASE_PATH, 'config')
DEFAULT_CONFIG_PATH = os.path.join(CONFIG_DIR, 'default_config.json')
USER_CONFIG_PATH = os.path.join(BASE_PATH, 'data.pkl')

# Ensure config directory exists
if not os.path.isdir(CONFIG_DIR):
    os.makedirs(CONFIG_DIR)

class ConfigManager:
    def __init__(self):
        self.default_config = self._load_default_config()
        self.user_config = self._load_user_config()
        self.config = {**self.default_config, **self.user_config}
    
    def _load_default_config(self):
        """Load default configuration"""
        # Default configuration
        default_config = {
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
            'ensemble_primary_stem': 'Vocals',
            'ensemble_secondary_stem': 'Instrumental',
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
        
        # Save default config to file
        if not os.path.isfile(DEFAULT_CONFIG_PATH):
            with open(DEFAULT_CONFIG_PATH, 'w') as f:
                json.dump(default_config, f, indent=4)
        
        return default_config
    
    def _load_user_config(self):
        """Load user configuration"""
        try:
            if os.path.isfile(USER_CONFIG_PATH):
                with open(USER_CONFIG_PATH, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            print(f"Error loading user config: {e}")
        
        return {}
    
    def get(self, key, default=None):
        """Get configuration value"""
        return self.config.get(key, default)
    
    def set(self, key, value):
        """Set configuration value"""
        self.config[key] = value
    
    def update(self, config_dict):
        """Update multiple configuration values"""
        self.config.update(config_dict)
    
    def save(self):
        """Save configuration to file"""
        try:
            with open(USER_CONFIG_PATH, 'wb') as f:
                pickle.dump(self.config, f)
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False
    
    def reset(self):
        """Reset configuration to default"""
        self.config = self.default_config.copy()
        return self.save()
    
    def validate(self):
        """Validate configuration"""
        # Ensure output directory exists
        export_path = self.config.get('export_path')
        if export_path and not os.path.isdir(export_path):
            try:
                os.makedirs(export_path)
            except Exception as e:
                print(f"Error creating export directory: {e}")
                return False
        
        # Validate device setting
        device_set = self.config.get('device_set')
        if device_set == 'cuda' and not os.environ.get('CUDA_VISIBLE_DEVICES'):
            print("CUDA not available, falling back to CPU")
            self.config['device_set'] = 'cpu'
        
        return True
    
    def get_process_config(self, process_method):
        """Get configuration for specific process method"""
        process_config = self.config.copy()
        
        # Add process-specific configuration
        if process_method == 'vr':
            process_config['model'] = process_config.get('vr_model')
        elif process_method == 'mdx':
            process_config['model'] = process_config.get('mdx_net_model')
        elif process_method == 'demucs':
            process_config['model'] = process_config.get('demucs_model')
        
        return process_config

# Create global config instance
config_manager = ConfigManager()

# Export config for easy access
def get_config():
    """Get global config instance"""
    return config_manager

def get_config_value(key, default=None):
    """Get config value"""
    return config_manager.get(key, default)

def set_config_value(key, value):
    """Set config value"""
    config_manager.set(key, value)
    return config_manager.save()

def update_config(config_dict):
    """Update config with dictionary"""
    config_manager.update(config_dict)
    return config_manager.save()

def reset_config():
    """Reset config to default"""
    return config_manager.reset()

def validate_config():
    """Validate config"""
    return config_manager.validate()
