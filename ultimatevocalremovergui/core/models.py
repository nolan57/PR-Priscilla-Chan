# Model management module
import os
import hashlib
import json
import math
import torch
import onnx
import yaml
from ml_collections import ConfigDict
from pathlib import Path

# Constants (will be imported from config)
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_PATH, 'models')
VR_MODELS_DIR = os.path.join(MODELS_DIR, 'VR_Models')
MDX_MODELS_DIR = os.path.join(MODELS_DIR, 'MDX_Net_Models')
DEMUCS_MODELS_DIR = os.path.join(MODELS_DIR, 'Demucs_Models')
DEMUCS_NEWER_REPO_DIR = os.path.join(DEMUCS_MODELS_DIR, 'v3_v4_repo')
MDX_MIXER_PATH = os.path.join(BASE_PATH, 'lib_v5', 'mixer.ckpt')

# Cache & Parameters
VR_HASH_DIR = os.path.join(VR_MODELS_DIR, 'model_data')
VR_HASH_JSON = os.path.join(VR_MODELS_DIR, 'model_data', 'model_data.json')
MDX_HASH_DIR = os.path.join(MDX_MODELS_DIR, 'model_data')
MDX_HASH_JSON = os.path.join(MDX_HASH_DIR, 'model_data.json')
MDX_C_CONFIG_PATH = os.path.join(MDX_HASH_DIR, 'mdx_c_configs')

DEMUCS_MODEL_NAME_SELECT = os.path.join(DEMUCS_MODELS_DIR, 'model_data', 'model_name_mapper.json')
MDX_MODEL_NAME_SELECT = os.path.join(MDX_MODELS_DIR, 'model_data', 'model_name_mapper.json')
ENSEMBLE_CACHE_DIR = os.path.join(BASE_PATH, 'gui_data', 'saved_ensembles')
SETTINGS_CACHE_DIR = os.path.join(BASE_PATH, 'gui_data', 'saved_settings')
VR_PARAM_DIR = os.path.join(BASE_PATH, 'lib_v5', 'vr_network', 'modelparams')

# Models
DENOISER_MODEL_PATH = os.path.join(VR_MODELS_DIR, 'UVR-DeNoise-Lite.pth')
DEVERBER_MODEL_PATH = os.path.join(VR_MODELS_DIR, 'UVR-DeEcho-DeReverb.pth')

# Imports
import sys
sys.path.insert(0, BASE_PATH)
from lib_v5.vr_network.model_param_init import ModelParameters

# Global variables
model_hash_table = {}

class ModelData:
    def __init__(self, model_name: str, 
                 selected_process_method='ensemble', 
                 is_secondary_model=False, 
                 primary_model_primary_stem=None, 
                 is_primary_model_primary_stem_only=False, 
                 is_primary_model_secondary_stem_only=False, 
                 is_pre_proc_model=False,
                 is_dry_check=False,
                 is_change_def=False,
                 is_get_hash_dir_only=False,
                 is_vocal_split_model=False,
                 config=None):
        
        # Use config object instead of root variables
        self.config = config or {}
        
        # Set default values from config
        device_set = self.config.get('device_set', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.DENOISER_MODEL = DENOISER_MODEL_PATH
        self.DEVERBER_MODEL = DEVERBER_MODEL_PATH
        self.is_deverb_vocals = self.config.get('is_deverb_vocals', False) if os.path.isfile(DEVERBER_MODEL_PATH) else False
        self.deverb_vocal_opt = self.config.get('deverb_vocal_opt', 'All Vocals')
        self.is_denoise_model = True if self.config.get('denoise_option', 'none') == 'denoise' and os.path.isfile(DENOISER_MODEL_PATH) else False
        self.is_gpu_conversion = 0 if self.config.get('is_gpu_conversion', True) else -1
        self.is_normalization = self.config.get('is_normalization', False)
        self.is_use_opencl = False
        self.is_primary_stem_only = self.config.get('is_primary_stem_only', False)
        self.is_secondary_stem_only = self.config.get('is_secondary_stem_only', False)
        self.is_denoise = True if not self.config.get('denoise_option', 'none') == 'none' else False
        self.is_mdx_c_seg_def = self.config.get('is_mdx_c_seg_def', False)
        mdx_batch_size = self.config.get('mdx_batch_size', 'def')
        self.mdx_batch_size = 1 if mdx_batch_size in ['def', 'Default'] else int(mdx_batch_size)
        self.mdxnet_stem_select = self.config.get('mdxnet_stems', 'vocals')
        overlap = self.config.get('overlap', 'default')
        self.overlap = float(overlap) if overlap not in ['default', 'Default'] else 0.25
        overlap_mdx = self.config.get('overlap_mdx', 'default')
        self.overlap_mdx = float(overlap_mdx) if overlap_mdx not in ['default', 'Default'] else 0.25
        self.overlap_mdx23 = int(float(self.config.get('overlap_mdx23', 8)))
        self.semitone_shift = float(self.config.get('semitone_shift', 0))
        self.is_pitch_change = False if self.semitone_shift == 0 else True
        self.is_match_frequency_pitch = self.config.get('is_match_frequency_pitch', False)
        self.is_mdx_ckpt = False
        self.is_mdx_c = False
        self.is_mdx_combine_stems = self.config.get('is_mdx23_combine_stems', False)
        self.mdx_c_configs = None
        self.mdx_model_stems = []
        self.mdx_dim_f_set = None
        self.mdx_dim_t_set = None
        self.mdx_stem_count = 1
        self.compensate = None
        self.mdx_n_fft_scale_set = None
        self.wav_type_set = self.config.get('wav_type_set', 'PCM_16')
        self.device_set = device_set.split(':')[-1].strip() if ':' in device_set else device_set
        self.mp3_bit_set = self.config.get('mp3_bit_set', '320')
        self.save_format = self.config.get('save_format', 'wav')
        self.is_invert_spec = self.config.get('is_invert_spec', False)
        self.is_mixer_mode = False
        self.demucs_stems = self.config.get('demucs_stems', 'vocals')
        self.is_demucs_combine_stems = self.config.get('is_demucs_combine_stems', False)
        self.demucs_source_list = []
        self.demucs_stem_count = 0
        self.mixer_path = MDX_MIXER_PATH
        self.model_name = model_name
        self.process_method = selected_process_method
        self.model_status = False if self.model_name == 'Choose Model' or self.model_name == 'No Model' else True
        self.primary_stem = 'Vocals'  # Default value
        self.secondary_stem = 'Instrumental'  # Default value
        self.primary_stem_native = None
        self.is_ensemble_mode = False
        self.ensemble_primary_stem = None
        self.ensemble_secondary_stem = None
        self.primary_model_primary_stem = primary_model_primary_stem
        self.is_secondary_model = True if is_vocal_split_model else is_secondary_model
        self.secondary_model = None
        self.secondary_model_scale = None
        self.demucs_4_stem_added_count = 0
        self.is_demucs_4_stem_secondaries = False
        self.is_4_stem_ensemble = False
        self.pre_proc_model = None
        self.pre_proc_model_activated = False
        self.is_pre_proc_model = is_pre_proc_model
        self.is_dry_check = is_dry_check
        self.model_samplerate = 44100
        self.model_capacity = 32, 128
        self.is_vr_51_model = False
        self.is_demucs_pre_proc_model_inst_mix = False
        self.manual_download_Button = None
        self.secondary_model_4_stem = []
        self.secondary_model_4_stem_scale = []
        self.secondary_model_4_stem_names = []
        self.secondary_model_4_stem_model_names_list = []
        self.all_models = []
        self.secondary_model_other = None
        self.secondary_model_scale_other = None
        self.secondary_model_bass = None
        self.secondary_model_scale_bass = None
        self.secondary_model_drums = None
        self.secondary_model_scale_drums = None
        self.is_multi_stem_ensemble = False
        self.is_karaoke = False
        self.is_bv_model = False
        self.bv_model_rebalance = 0
        self.is_sec_bv_rebalance = False
        self.is_change_def = is_change_def
        self.model_hash_dir = None
        self.is_get_hash_dir_only = is_get_hash_dir_only
        self.is_secondary_model_activated = False
        self.vocal_split_model = None
        self.is_vocal_split_model = is_vocal_split_model
        self.is_vocal_split_model_activated = False
        self.is_save_inst_vocal_splitter = self.config.get('is_save_inst_set_vocal_splitter', False)
        self.is_inst_only_voc_splitter = False  # Will be set based on config
        self.is_save_vocal_only = False  # Will be set based on config

        if selected_process_method == 'ensemble':
            # Handle ensemble mode
            if '|' in model_name:
                self.process_method, _, self.model_name = model_name.partition('|')
            self.model_and_process_tag = model_name
            self.ensemble_primary_stem = self.config.get('ensemble_primary_stem', 'Vocals')
            self.ensemble_secondary_stem = self.config.get('ensemble_secondary_stem', 'Instrumental')
            
            is_not_secondary_or_pre_proc = not is_secondary_model and not is_pre_proc_model
            self.is_ensemble_mode = is_not_secondary_or_pre_proc
            
            ensemble_main_stem = self.config.get('ensemble_main_stem', 'Vocals/Instrumental')
            if ensemble_main_stem == '4-Stem Ensemble':
                self.is_4_stem_ensemble = self.is_ensemble_mode
            elif ensemble_main_stem == 'Multi-Stem Ensemble':
                self.is_multi_stem_ensemble = True

            is_not_vocal_stem = self.ensemble_primary_stem != 'Vocals'
            self.pre_proc_model_activated = self.config.get('is_demucs_pre_proc_model_activate', False) if is_not_vocal_stem else False

        if self.process_method == 'vr':
            self.is_secondary_model_activated = self.config.get('vr_is_secondary_model_activate', False) if not is_secondary_model else False
            self.aggression_setting = float(int(self.config.get('aggression_setting', 10))/100)
            self.is_tta = self.config.get('is_tta', False)
            self.is_post_process = self.config.get('is_post_process', False)
            self.window_size = int(self.config.get('window_size', 512))
            batch_size = self.config.get('batch_size', 'def')
            self.batch_size = 1 if batch_size in ['def', 'Default'] else int(batch_size)
            self.crop_size = int(self.config.get('crop_size', 1024))
            self.is_high_end_process = 'mirroring' if self.config.get('is_high_end_process', False) else 'None'
            self.post_process_threshold = float(self.config.get('post_process_threshold', 0.5))
            self.model_capacity = 32, 128
            # Check if model_name is a full path
            if os.path.isabs(self.model_name) and os.path.exists(self.model_name):
                self.model_path = self.model_name
            else:
                self.model_path = os.path.join(VR_MODELS_DIR, f"{self.model_name}.pth")
            print(f"DEBUG: model_path = {self.model_path}")
            print(f"DEBUG: os.path.exists(model_path) = {os.path.exists(self.model_path)}")
            self.get_model_hash()
            print(f"DEBUG: model_hash = {self.model_hash}")
            print(f"DEBUG: model_status = {self.model_status}")
            if self.model_hash:
                self.model_hash_dir = os.path.join(VR_HASH_DIR, f"{self.model_hash}.json")
                if is_change_def:
                    self.model_data = self.change_model_data()
                else:
                    # Load model data from JSON
                    self.model_data = self.get_model_data(VR_HASH_DIR)
                print(f"DEBUG: model_data = {self.model_data}")
                if self.model_data:
                    vr_model_param = os.path.join(VR_PARAM_DIR, "{}.json".format(self.model_data["vr_model_param"]))
                    self.primary_stem = self.model_data["primary_stem"]
                    self.secondary_stem = self._secondary_stem(self.primary_stem)
                    print(f"DEBUG: primary_stem = {self.primary_stem}")
                    print(f"DEBUG: secondary_stem = {self.secondary_stem}")
                    self.vr_model_param = ModelParameters(vr_model_param)
                    self.model_samplerate = self.vr_model_param.param['sr']
                    self.primary_stem_native = self.primary_stem
                    if "nout" in self.model_data and "nout_lstm" in self.model_data:
                        self.model_capacity = self.model_data["nout"], self.model_data["nout_lstm"]
                        self.is_vr_51_model = True
                    self.check_if_karaokee_model()
                else:
                    self.model_status = False
                    print(f"DEBUG: model_data is None, setting model_status to False")
            else:
                print(f"DEBUG: model_hash is None, skipping model data loading")

        if self.process_method == 'mdx':
            self.is_secondary_model_activated = self.config.get('mdx_is_secondary_model_activate', False) if not is_secondary_model else False
            self.margin = int(self.config.get('margin', 10))
            self.chunks = 0
            self.mdx_segment_size = int(self.config.get('mdx_segment_size', 256))
            self.get_mdx_model_path()
            self.get_model_hash()
            if self.model_hash:
                self.model_hash_dir = os.path.join(MDX_HASH_DIR, f"{self.model_hash}.json")
                if is_change_def:
                    self.model_data = self.change_model_data()
                else:
                    self.model_data = self.get_model_data(MDX_HASH_DIR)
                if self.model_data:
                    if "config_yaml" in self.model_data:
                        self.is_mdx_c = True
                        config_path = os.path.join(MDX_C_CONFIG_PATH, self.model_data["config_yaml"])
                        if os.path.isfile(config_path):
                            with open(config_path) as f:
                                config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))
                            self.mdx_c_configs = config
                            if self.mdx_c_configs.training.target_instrument:
                                # Use target_instrument as the primary stem
                                target = self.mdx_c_configs.training.target_instrument
                                self.mdx_model_stems = [target]
                                self.primary_stem = target
                            else:
                                # If no specific target_instrument, use all instruments
                                self.mdx_model_stems = self.mdx_c_configs.training.instruments
                                self.mdx_stem_count = len(self.mdx_model_stems)
                                # Set primary stem based on stem count
                                if self.mdx_stem_count == 2:
                                    self.primary_stem = self.mdx_model_stems[0]
                                else:
                                    self.primary_stem = self.mdxnet_stem_select
                                # Update mdxnet_stem_select based on ensemble mode
                                if self.is_ensemble_mode:
                                    self.mdxnet_stem_select = self.ensemble_primary_stem
                        else:
                            self.model_status = False
                    else:
                        self.compensate = self.model_data["compensate"] if self.config.get('compensate', 'auto') == 'auto' else float(self.config.get('compensate', 1.0))
                        self.mdx_dim_f_set = self.model_data["mdx_dim_f_set"]
                        self.mdx_dim_t_set = self.model_data["mdx_dim_t_set"]
                        self.mdx_n_fft_scale_set = self.model_data["mdx_n_fft_scale_set"]
                        self.primary_stem = self.model_data["primary_stem"]
                        self.primary_stem_native = self.model_data["primary_stem"]
                        self.check_if_karaokee_model()
                    self.secondary_stem = self._secondary_stem(self.primary_stem)
                else:
                    self.model_status = False

        if self.process_method == 'demucs':
            self.is_secondary_model_activated = self.config.get('demucs_is_secondary_model_activate', False) if not is_secondary_model else False
            if not self.is_ensemble_mode:
                is_not_vocal_stem = self.demucs_stems not in ['vocals', 'instrumental']
                self.pre_proc_model_activated = self.config.get('is_demucs_pre_proc_model_activate', False) if is_not_vocal_stem else False
            self.margin_demucs = int(self.config.get('margin_demucs', 10))
            self.chunks_demucs = 0
            self.shifts = int(self.config.get('shifts', 1))
            self.is_split_mode = self.config.get('is_split_mode', False)
            self.segment = self.config.get('segment', '10')
            self.is_chunk_demucs = self.config.get('is_chunk_demucs', False)
            self.is_primary_stem_only = self.config.get('is_primary_stem_only_Demucs', False) if not self.is_ensemble_mode else self.is_primary_stem_only
            self.is_secondary_stem_only = self.config.get('is_secondary_stem_only_Demucs', False) if not self.is_ensemble_mode else self.is_secondary_stem_only
            self.get_demucs_model_data()
            self.get_demucs_model_path()
            
        if self.model_status:
            self.model_basename = os.path.splitext(os.path.basename(self.model_path))[0]
        else:
            self.model_basename = None
            
        self.pre_proc_model_activated = self.pre_proc_model_activated if not self.is_secondary_model else False
        
        self.is_primary_model_primary_stem_only = is_primary_model_primary_stem_only
        self.is_primary_model_secondary_stem_only = is_primary_model_secondary_stem_only

        # Handle secondary models
        is_secondary_activated_and_status = self.is_secondary_model_activated and self.model_status
        is_demucs = self.process_method == 'demucs'
        is_all_stems = self.demucs_stems == 'All Stems'
        is_valid_ensemble = not self.is_ensemble_mode and is_all_stems and is_demucs
        is_multi_stem_ensemble_demucs = self.is_multi_stem_ensemble and is_demucs

        if is_secondary_activated_and_status:
            if is_valid_ensemble or self.is_4_stem_ensemble or is_multi_stem_ensemble_demucs:
                # Handle 4-stem secondary models
                for key in ['vocals', 'drums', 'bass', 'other']:
                    self.secondary_model_data(key)
                    if self.secondary_model:
                        self.secondary_model_4_stem.append(self.secondary_model)
                        self.secondary_model_4_stem_scale.append(self.secondary_model_scale)
                        self.secondary_model_4_stem_names.append(key)
                
                self.demucs_4_stem_added_count = sum(1 for i in self.secondary_model_4_stem if i is not None)
                self.is_secondary_model_activated = any(i is not None for i in self.secondary_model_4_stem)
                self.demucs_4_stem_added_count -= 1 if self.is_secondary_model_activated else 0
                
                if self.is_secondary_model_activated:
                    self.secondary_model_4_stem_model_names_list = [i.model_basename if i is not None else None for i in self.secondary_model_4_stem]
                    self.is_demucs_4_stem_secondaries = True
            else:
                primary_stem = self.ensemble_primary_stem if self.is_ensemble_mode and is_demucs else self.primary_stem
                self.secondary_model_data(primary_stem)

        if self.process_method == 'demucs' and not is_secondary_model:
            if self.demucs_stem_count >= 3 and self.pre_proc_model_activated:
                # Handle pre-process model
                self.pre_proc_model = None  # Will be set based on config
                self.pre_proc_model_activated = True if self.pre_proc_model else False
                self.is_demucs_pre_proc_model_inst_mix = self.config.get('is_demucs_pre_proc_model_inst_mix', False) if self.pre_proc_model else False

        if self.is_vocal_split_model and self.model_status:
            self.is_secondary_model_activated = False
            if self.is_bv_model:
                primary = 'BV Vocals' if self.primary_stem_native == 'Vocals' else 'Lead Vocals'
            else:
                primary = 'Lead Vocals' if self.primary_stem_native == 'Vocals' else 'BV Vocals'
            self.primary_stem, self.secondary_stem = primary, self._secondary_stem(primary)
            
        # Set vocal splitter flags
        self.vocal_splitter_model_data()

    def vocal_splitter_model_data(self):
        """Initialize vocal splitter model data"""
        if not self.is_secondary_model and self.model_status:
            # Set vocal splitter based on config
            set_vocal_splitter = self.config.get('set_vocal_splitter', 'No Model')
            is_set_vocal_splitter = self.config.get('is_set_vocal_splitter', False)
            if set_vocal_splitter != 'No Model' and is_set_vocal_splitter:
                # Create vocal splitter model
                self.vocal_split_model = None  # Will be set based on config
                self.is_vocal_split_model_activated = True if self.vocal_split_model else False
                
                if self.vocal_split_model and hasattr(self.vocal_split_model, 'bv_model_rebalance'):
                    if self.vocal_split_model.bv_model_rebalance:
                        self.is_sec_bv_rebalance = True

    def secondary_model_data(self, primary_stem):
        """Initialize secondary model data"""
        # This method will be implemented based on config
        secondary_model_name = None
        secondary_model_scale = None
        
        # Get secondary model based on primary stem
        if primary_stem in ['Vocals', 'Instrumental']:
            secondary_model_name = self.config.get('vr_voc_inst_secondary_model', 'No Model')
            secondary_model_scale = self.config.get('vr_voc_inst_secondary_model_scale', 0.5)
        elif primary_stem == 'Other':
            secondary_model_name = self.config.get('vr_other_secondary_model', 'No Model')
            secondary_model_scale = self.config.get('vr_other_secondary_model_scale', 0.5)
        elif primary_stem == 'Bass':
            secondary_model_name = self.config.get('vr_bass_secondary_model', 'No Model')
            secondary_model_scale = self.config.get('vr_bass_secondary_model_scale', 0.5)
        elif primary_stem == 'Drums':
            secondary_model_name = self.config.get('vr_drums_secondary_model', 'No Model')
            secondary_model_scale = self.config.get('vr_drums_secondary_model_scale', 0.5)

        if secondary_model_name and secondary_model_name != 'No Model':
            # Create secondary model
            self.secondary_model = None  # Will be implemented
            self.secondary_model_scale = float(secondary_model_scale)
            self.is_secondary_model_activated = True if self.secondary_model else False
            if self.secondary_model:
                if self.secondary_model.model_basename == self.model_basename:
                    self.is_secondary_model_activated = False
        else:
            self.secondary_model = None
            self.secondary_model_scale = None
            self.is_secondary_model_activated = False

    def check_if_karaokee_model(self):
        """Check if model is a karaoke or backing vocal model"""
        if self.model_data:
            if "is_karaoke" in self.model_data:
                self.is_karaoke = self.model_data["is_karaoke"]
            if "is_bv_model" in self.model_data:
                self.is_bv_model = self.model_data["is_bv_model"]
            if "is_bv_model_rebal" in self.model_data and self.is_bv_model:
                self.bv_model_rebalance = self.model_data["is_bv_model_rebal"]

    def get_mdx_model_path(self):
        """Get MDX model path"""
        if self.model_name.endswith('.ckpt'):
            self.is_mdx_ckpt = True

        ext = '' if self.is_mdx_ckpt else '.onnx'
        
        # Load model name mapper
        mdx_name_select_mapper = {}
        mapper_path = os.path.join(MDX_MODELS_DIR, 'model_data', 'model_name_mapper.json')
        if os.path.isfile(mapper_path):
            with open(mapper_path, 'r') as f:
                mdx_name_select_mapper = json.load(f)
        
        for file_name, chosen_mdx_model in mdx_name_select_mapper.items():
            if self.model_name in chosen_mdx_model:
                if file_name.endswith('.ckpt'):
                    ext = ''
                self.model_path = os.path.join(MDX_MODELS_DIR, f"{file_name}{ext}")
                break
        else:
            self.model_path = os.path.join(MDX_MODELS_DIR, f"{self.model_name}{ext}")
            
        self.mixer_path = os.path.join(MDX_MODELS_DIR, f"mixer_val.ckpt")

    def get_demucs_model_path(self):
        """Get Demucs model path"""
        demucs_newer = self.demucs_version in {'v3', 'v4'}
        demucs_model_dir = DEMUCS_NEWER_REPO_DIR if demucs_newer else DEMUCS_MODELS_DIR
        
        # Load model name mapper
        demucs_name_select_mapper = {}
        mapper_path = os.path.join(DEMUCS_MODELS_DIR, 'model_data', 'model_name_mapper.json')
        if os.path.isfile(mapper_path):
            with open(mapper_path, 'r') as f:
                demucs_name_select_mapper = json.load(f)
        
        for file_name, chosen_model in demucs_name_select_mapper.items():
            if self.model_name == chosen_model:
                self.model_path = os.path.join(demucs_model_dir, file_name)
                break
        else:
            self.model_path = os.path.join(DEMUCS_NEWER_REPO_DIR, f'{self.model_name}.yaml')

    def get_demucs_model_data(self):
        """Get Demucs model data"""
        self.demucs_version = 'v4'

        # Map model versions
        demucs_version_mapper = {
            'v3': ['v3'],
            'v4': ['v4']
        }
        
        for key, values in demucs_version_mapper.items():
            for value in values:
                if value in self.model_name:
                    self.demucs_version = key
                    break

        if 'uvr' in self.model_name.lower():
            self.demucs_source_list = ['vocals', 'instrumental']
            self.demucs_stem_count = 2
        else:
            self.demucs_source_list = ['vocals', 'drums', 'bass', 'other']
            self.demucs_stem_count = 4

        if not self.is_ensemble_mode:
            self.primary_stem = 'Primary Stem' if self.demucs_stems == 'All Stems' else self.demucs_stems
            self.secondary_stem = self._secondary_stem(self.primary_stem)

    def get_model_data(self, model_hash_dir):
        """Get model data from hash directory"""
        model_settings_json = os.path.join(model_hash_dir, f"{self.model_hash}.json")

        if os.path.isfile(model_settings_json):
            with open(model_settings_json, 'r') as json_file:
                return json.load(json_file)
        else:
            # Try to load from hash mapper
            hash_mapper = {}
            mapper_path = os.path.join(model_hash_dir, 'model_data.json')
            if os.path.isfile(mapper_path):
                with open(mapper_path, 'r') as f:
                    hash_mapper = json.load(f)
            
            for hash_key, settings in hash_mapper.items():
                if self.model_hash in hash_key:
                    return settings

            return self.get_model_data_from_config()

    def change_model_data(self):
        """Change model data"""
        if self.is_get_hash_dir_only:
            return None
        else:
            return self.get_model_data_from_config()

    def get_model_data_from_config(self):
        """Get model data from config"""
        if self.is_dry_check:
            return None
        
        # Return default model data based on process method
        if self.process_method == 'vr':
            return {
                'vr_model_param': 'default',
                'primary_stem': 'Vocals',
                'nout': 32,
                'nout_lstm': 128
            }
        elif self.process_method == 'mdx':
            return {
                'compensate': 1.035,
                'mdx_dim_f_set': 32,
                'mdx_dim_t_set': 9,
                'mdx_n_fft_scale_set': 6144,
                'primary_stem': 'Vocals'
            }
        return None

    def get_model_hash(self):
        """Get model hash"""
        self.model_hash = None
        
        if not os.path.isfile(self.model_path):
            self.model_status = False
            self.model_hash = None
        else:
            if model_hash_table:
                for (key, value) in model_hash_table.items():
                    if self.model_path == key:
                        self.model_hash = value
                        break
                    
            if not self.model_hash:
                try:
                    with open(self.model_path, 'rb') as f:
                        f.seek(- 10000 * 1024, 2)
                        self.model_hash = hashlib.md5(f.read()).hexdigest()
                except:
                    # If seeking fails, read entire file
                    with open(self.model_path, 'rb') as f:
                        self.model_hash = hashlib.md5(f.read()).hexdigest()
                    
                table_entry = {self.model_path: self.model_hash}
                model_hash_table.update(table_entry)

    def _secondary_stem(self, primary_stem):
        """Get secondary stem based on primary stem"""
        stem_map = {
            'Vocals': 'Instrumental',
            'Instrumental': 'Vocals',
            'Lead Vocals': 'BV Vocals',
            'BV Vocals': 'Lead Vocals'
        }
        return stem_map.get(primary_stem, 'Instrumental')
