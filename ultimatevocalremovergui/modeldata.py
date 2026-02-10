import os
from pathlib import Path
from typing import Optional

from lib_v5.vr_network.model_param_init import ModelParameters
from gui_data.constants import *

# Minimal GUI-free ModelData for standalone use by `separate.py`.
# This mirrors the attributes `separate.py` expects but does not depend on the
# `root` GUI object in `UVR.py`.

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_PATH, 'models')
VR_MODELS_DIR = os.path.join(MODELS_DIR, 'VR_Models')
MDX_MODELS_DIR = os.path.join(MODELS_DIR, 'MDX_Net_Models')
DEMUCS_MODELS_DIR = os.path.join(MODELS_DIR, 'Demucs_Models')
VR_PARAM_DIR = os.path.join(BASE_PATH, 'lib_v5', 'vr_network', 'modelparams')

DEFAULT_WAV_TYPE = 'PCM_16'


class ModelData:
    def __init__(self,
                 model_name: str,
                 selected_process_method: str = ENSEMBLE_MODE,
                 is_secondary_model: bool = False,
                 primary_model_primary_stem: Optional[str] = None,
                 is_primary_model_primary_stem_only: bool = False,
                 is_primary_model_secondary_stem_only: bool = False,
                 is_pre_proc_model: bool = False,
                 is_dry_check: bool = False,
                 is_change_def: bool = False,
                 is_get_hash_dir_only: bool = False,
                 is_vocal_split_model: bool = False):

        self.model_name = model_name
        self.process_method = selected_process_method
        self.is_secondary_model = is_secondary_model
        self.primary_model_primary_stem = primary_model_primary_stem
        self.is_primary_model_primary_stem_only = is_primary_model_primary_stem_only
        self.is_primary_model_secondary_stem_only = is_primary_model_secondary_stem_only
        self.is_pre_proc_model = is_pre_proc_model
        self.is_dry_check = is_dry_check
        self.is_change_def = is_change_def
        self.is_get_hash_dir_only = is_get_hash_dir_only
        self.is_vocal_split_model = is_vocal_split_model

        # basic defaults used by separate.py
        self.DENOISER_MODEL = os.path.join(VR_MODELS_DIR, 'UVR-DeNoise-Lite.pth')
        self.DEVERBER_MODEL = os.path.join(VR_MODELS_DIR, 'UVR-DeEcho-DeReverb.pth')
        self.deverb_vocal_opt = 'ALL'
        self.is_denoise_model = False
        self.is_gpu_conversion = -1
        self.is_normalization = True
        self.is_use_opencl = False
        self.is_primary_stem_only = False
        self.is_secondary_stem_only = False
        self.is_mdx_c = False
        self.is_mdx_ckpt = False
        self.is_mdx_combine_stems = False
        self.mdx_c_configs = None
        self.mdxnet_stem_select = ALL_STEMS
        self.overlap = 0.25
        self.overlap_mdx = DEFAULT
        self.overlap_mdx23 = 8
        self.semitone_shift = 0
        self.is_pitch_change = False
        self.is_match_frequency_pitch = False
        self.mdx_batch_size = 1
        self.mdx_segment_size = 256
        self.mdx_dim_f_set = None
        self.mdx_dim_t_set = None
        self.mdx_n_fft_scale_set = None
        self.mdx_stem_count = 1
        self.compensate = 1.0
        self.wav_type_set = DEFAULT_WAV_TYPE
        self.device_set = DEFAULT
        self.mp3_bit_set = '192k'
        self.save_format = WAV
        self.is_invert_spec = False
        self.is_mixer_mode = False
        self.demucs_stems = ALL_STEMS
        self.is_demucs_combine_stems = False
        self.demucs_source_list = []
        self.demucs_stem_count = 2
        self.mixer_path = os.path.join(MODELS_DIR, 'MX.ckpt')

        # VR / general
        self.model_samplerate = 44100
        self.model_capacity = (32, 128)
        self.is_vr_51_model = False
        self.vr_model_param = None
        if self.process_method == VR_ARCH_TYPE:
            # Try to load a default vr param if available
            default_param = os.path.join(VR_PARAM_DIR, '4band_v3.json')
            if os.path.isfile(default_param):
                try:
                    self.vr_model_param = ModelParameters(default_param)
                    self.model_samplerate = self.vr_model_param.param.get('sr', 44100)
                except Exception:
                    self.vr_model_param = None

        # MDX
        if self.process_method == MDX_ARCH_TYPE:
            self.is_mdx_ckpt = self.model_name.endswith(CKPT)
            self.is_mdx_c = False

        # DEMUCS
        if self.process_method == DEMUCS_ARCH_TYPE:
            self.demucs_version = DEMUCS_V4
            self.demucs_source_list = DEMUCS_4_SOURCE
            self.demucs_source_map = DEMUCS_4_SOURCE_MAPPER

        # secondary/ensemble defaults
        self.is_secondary_model_activated = False
        self.secondary_model = None
        self.secondary_model_scale = None
        self.secondary_model_4_stem = []
        self.secondary_model_4_stem_scale = []
        self.secondary_model_4_stem_names = []
        self.secondary_model_4_stem_model_names_list = []
        self.all_models = []
        self.pre_proc_model = None
        self.pre_proc_model_activated = False
        self.is_ensemble_mode = False
        self.ensemble_primary_stem = None
        self.ensemble_secondary_stem = None
        self.is_multi_stem_ensemble = False
        self.is_karaoke = False
        self.is_bv_model = False
        self.bv_model_rebalance = 0
        self.is_sec_bv_rebalance = False
        self.vocal_split_model = None
        # Primary/secondary stem defaults (used heavily by separate.py)
        # If a primary model stem was provided use it, otherwise default to VOCAL_STEM
        self.primary_stem = self.primary_model_primary_stem if self.primary_model_primary_stem else VOCAL_STEM
        self.primary_stem_native = self.primary_stem
        # secondary_stem() is provided by gui_data.constants and maps vocal<->inst etc.
        try:
            self.secondary_stem = secondary_stem(self.primary_stem)
        except Exception:
            # fallback: sensible default
            self.secondary_stem = INST_STEM
        # Ensure ensemble stem defaults follow primary/secondary
        if not self.ensemble_primary_stem:
            self.ensemble_primary_stem = self.primary_stem
        if not self.ensemble_secondary_stem:
            self.ensemble_secondary_stem = self.secondary_stem
        # Vocal-split / save options
        self.is_save_inst_vocal_splitter = False
        self.is_save_vocal_only = False
        # Demucs / MDX defaults
        self.is_chunk_demucs = False
        self.chunks_demucs = CHUNKS[0] if 'CHUNKS' in globals() else DEFAULT
        self.margin_demucs = 44100
        self.shifts = 2
        self.segment = DEMUCS_SEGMENTS[0] if 'DEMUCS_SEGMENTS' in globals() else DEF_OPT
        self.is_split_mode = True
        # MDX defaults used by separate.py
        self.chunks = CHUNKS[0] if 'CHUNKS' in globals() else DEFAULT
        self.margin = 44100
        self.is_denoise = False
        self.is_denoise_model = False
        self.is_mdx_c_seg_def = False
        # demucs preproc / vocal-split defaults referenced elsewhere
        self.is_demucs_pre_proc_model_inst_mix = False
        self.is_inst_only_voc_splitter = False
        self.is_high_end_process = 'None'
        self.is_tta = False
        
        # Additional attributes used by separate.py
        self.model_status = True if self.model_name not in (CHOOSE_MODEL, NO_MODEL) else False
        self.model_path = self._resolve_model_path()
        self.model_basename = os.path.splitext(os.path.basename(self.model_path if self.model_path else self.model_name))[0]
        
        # Additional VR-specific attributes
        if self.process_method == VR_ARCH_TYPE:
            self.aggression_setting = 0.0
            self.is_post_process = False
            self.window_size = 512
            self.batch_size = 1
            self.crop_size = 512
            self.post_process_threshold = 0.2
            self.is_4_stem_ensemble = False
            self.pre_proc_model = None
            self.demucs_4_stem_added_count = 0
            self.is_demucs_4_stem_secondaries = False
            self.is_deverb_vocals = False
            self.mdx_model_stems = []
            self.primary_model_primary_stem = self.primary_stem
            self.model_hash_dir = None
            self.model_hash = None
            self.model_data = None

        # Additional MDX-specific attributes
        if self.process_method == MDX_ARCH_TYPE:
            self.mdxnet_stem_select = ALL_STEMS
            self.mdx_model_stems = []
            self.is_4_stem_ensemble = False
            self.is_mdx_c_seg_def = False
            self.primary_model_primary_stem = self.primary_stem

        # Additional DEMUCS-specific attributes
        if self.process_method == DEMUCS_ARCH_TYPE:
            self.demucs_version = DEMUCS_V4
            self.demucs_source_map = DEMUCS_4_SOURCE_MAPPER
            self.is_chunk_demucs = False
            self.is_4_stem_ensemble = False
            self.pre_proc_model_activated = False
            self.is_demucs_pre_proc_model_inst_mix = False

        # Ensemble-related attributes
        self.is_4_stem_ensemble = False
        self.pre_proc_model_activated = False
        self.is_vocal_split_model_activated = False
        self.manual_download_Button = None
        self.secondary_model_other = None
        self.secondary_model_scale_other = None
        self.secondary_model_bass = None
        self.secondary_model_scale_bass = None
        self.secondary_model_drums = None
        self.secondary_model_scale_drums = None
        self.model_hash_dir = None

    def _resolve_model_path(self):
        # Try to construct a reasonable model path from the model name and process method
        if os.path.isabs(self.model_name) and os.path.exists(self.model_name):
            return self.model_name

        if self.process_method == VR_ARCH_TYPE:
            path = os.path.join(VR_MODELS_DIR, self.model_name)
            if os.path.exists(path):
                return path
            # try pth extension
            if not os.path.splitext(path)[1] and os.path.exists(path + '.pth'):
                return path + '.pth'

        if self.process_method == MDX_ARCH_TYPE:
            path = os.path.join(MDX_MODELS_DIR, self.model_name)
            if os.path.exists(path):
                return path
            if not os.path.splitext(path)[1] and os.path.exists(path + '.onnx'):
                return path + '.onnx'

        if self.process_method == DEMUCS_ARCH_TYPE:
            # accept ckpt or yaml
            path = os.path.join(DEMUCS_MODELS_DIR, self.model_name)
            if os.path.exists(path):
                return path
            if not os.path.splitext(path)[1] and os.path.exists(path + '.ckpt'):
                return path + '.ckpt'
            if not os.path.splitext(path)[1] and os.path.exists(path + '.yaml'):
                return path + '.yaml'

        # fallback to model name (as-is)
        return self.model_name