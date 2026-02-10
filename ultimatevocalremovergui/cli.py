#!/usr/bin/env python3
# Command line interface for Ultimate Vocal Remover
import os
import sys
import argparse
import time
import logging
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Imports
from core.models import ModelData
from core.ensemble import Ensembler
from core.utils import (
    process_input_selections, verify_audio, create_sample, 
    process_storage_check, clear_gpu_cache, remove_temps
)
from core.config import get_config, validate_config, update_config
from separate import SeperateDemucs, SeperateMDX, SeperateMDXC, SeperateVR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CLI:
    def __init__(self):
        self.config = get_config().config
        self.input_paths = []
        self.export_path = self.config.get('export_path')
        self.process_method = self.config.get('chosen_process_method')
        self.model_name = None
        self.is_ensemble = False
        self.ensemble_models = []
    
    def parse_args(self):
        """Parse command line arguments"""
        parser = argparse.ArgumentParser(
            description='Ultimate Vocal Remover CLI',
            formatter_class=argparse.RawTextHelpFormatter
        )
        
        # Input/output arguments
        parser.add_argument('input', nargs='+', help='Input audio files or directories')
        parser.add_argument('-o', '--output', help='Output directory')
        
        # Processing arguments
        parser.add_argument('-m', '--method', choices=['vr', 'mdx', 'demucs', 'ensemble'], help='Processing method')
        parser.add_argument('-t', '--model', help='Model name')
        parser.add_argument('-d', '--device', choices=['cpu', 'cuda'], help='Device to use')
        
        # VR-specific arguments
        parser.add_argument('--aggression', type=int, help='Aggression setting (0-100)')
        parser.add_argument('--window-size', type=int, help='Window size')
        parser.add_argument('--batch-size', type=int, help='Batch size')
        parser.add_argument('--crop-size', type=int, help='Crop size')
        parser.add_argument('--tta', action='store_true', help='Enable TTA')
        parser.add_argument('--post-process', action='store_true', help='Enable post-processing')
        parser.add_argument('--high-end-process', action='store_true', help='Enable high-end processing')
        
        # MDX-specific arguments
        parser.add_argument('--margin', type=int, help='Margin')
        parser.add_argument('--compensate', type=float, help='Volume compensation')
        parser.add_argument('--chunks', type=int, help='Chunks')
        parser.add_argument('--segment-size', type=int, help='Segment size')
        
        # Demucs-specific arguments
        parser.add_argument('--segment', type=float, help='Segment duration')
        parser.add_argument('--overlap', type=float, help='Overlap')
        parser.add_argument('--shifts', type=int, help='Shifts')
        parser.add_argument('--demucs-stems', choices=['vocals', 'drums', 'bass', 'other', 'all'], help='Demucs stems')
        
        # Ensemble arguments
        parser.add_argument('--ensemble-models', nargs='+', help='Models to ensemble')
        parser.add_argument('--ensemble-algorithm', choices=['mean', 'median', 'sum'], help='Ensemble algorithm')
        
        # General arguments
        parser.add_argument('--export-format', choices=['wav', 'mp3', 'flac'], help='Export format')
        parser.add_argument('--mp3-bitrate', choices=['128', '192', '256', '320'], help='MP3 bitrate')
        parser.add_argument('--normalize', action='store_true', help='Normalize output')
        parser.add_argument('--primary-only', action='store_true', help='Save only primary stem')
        parser.add_argument('--secondary-only', action='store_true', help='Save only secondary stem')
        parser.add_argument('--sample', type=int, help='Create sample of specified duration')
        parser.add_argument('--gpu', action='store_true', help='Force GPU usage')
        parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
        parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
        
        return parser.parse_args()
    
    def configure(self, args):
        """Configure CLI based on arguments"""
        # Set input paths
        self.input_paths = args.input
        
        # Set output directory
        if args.output:
            self.export_path = args.output
            self.config['export_path'] = self.export_path
        
        # Set processing method
        if args.method:
            self.process_method = args.method
            self.config['chosen_process_method'] = self.process_method
        
        # Set model
        if args.model:
            self.model_name = args.model
            if self.process_method == 'vr':
                self.config['vr_model'] = self.model_name
            elif self.process_method == 'mdx':
                self.config['mdx_net_model'] = self.model_name
            elif self.process_method == 'demucs':
                self.config['demucs_model'] = self.model_name
        
        # Set device
        if args.device:
            self.config['device_set'] = args.device
        elif args.gpu:
            self.config['device_set'] = 'cuda'
        elif args.cpu:
            self.config['device_set'] = 'cpu'
        
        # Set VR-specific arguments
        if args.aggression:
            self.config['aggression_setting'] = str(args.aggression)
        if args.window_size:
            self.config['window_size'] = str(args.window_size)
        if args.batch_size:
            self.config['batch_size'] = str(args.batch_size)
        if args.crop_size:
            self.config['crop_size'] = str(args.crop_size)
        if args.tta:
            self.config['is_tta'] = args.tta
        if args.post_process:
            self.config['is_post_process'] = args.post_process
        if args.high_end_process:
            self.config['is_high_end_process'] = args.high_end_process
        
        # Set MDX-specific arguments
        if args.margin:
            self.config['margin'] = str(args.margin)
        if args.compensate:
            self.config['compensate'] = str(args.compensate)
        if args.chunks:
            self.config['chunks'] = str(args.chunks)
        if args.segment_size:
            self.config['mdx_segment_size'] = str(args.segment_size)
        
        # Set Demucs-specific arguments
        if args.segment:
            self.config['segment'] = str(args.segment)
        if args.overlap:
            self.config['overlap'] = str(args.overlap)
        if args.shifts:
            self.config['shifts'] = str(args.shifts)
        if args.demucs_stems:
            self.config['demucs_stems'] = args.demucs_stems
        
        # Set ensemble arguments
        if args.ensemble_models:
            self.is_ensemble = True
            self.ensemble_models = args.ensemble_models
            self.config['ensemble_models'] = self.ensemble_models
        if args.ensemble_algorithm:
            self.config['ensemble_type'] = args.ensemble_algorithm
        
        # Set general arguments
        if args.export_format:
            self.config['save_format'] = args.export_format
        if args.mp3_bitrate:
            self.config['mp3_bit_set'] = args.mp3_bitrate
        if args.normalize:
            self.config['is_normalization'] = args.normalize
        if args.primary_only:
            self.config['is_primary_stem_only'] = args.primary_only
        if args.secondary_only:
            self.config['is_secondary_stem_only'] = args.secondary_only
        
        # Set verbose mode
        if args.verbose:
            logging.basicConfig(level=logging.DEBUG)
        
        # Validate configuration
        validate_config()
    
    def process(self):
        """Process input files"""
        # Process input selections
        input_files = process_input_selections(self.input_paths, self.config.get('is_accept_any_input', False))
        
        if not input_files:
            logger.error('No valid input files found')
            return False
        
        # Check storage
        if not process_storage_check():
            logger.error('Insufficient storage space')
            return False
        
        logger.info(f'Found {len(input_files)} input files')
        logger.info(f'Processing method: {self.process_method}')
        logger.info(f'Output directory: {self.export_path}')
        
        # Process each file
        for i, input_file in enumerate(input_files, 1):
            logger.info(f'Processing file {i}/{len(input_files)}: {os.path.basename(input_file)}')
            
            try:
                # Create sample if requested
                if self.config.get('sample'):
                    input_file = create_sample(input_file, self.config.get('sample'))
                
                # Verify audio
                if not verify_audio(input_file):
                    logger.error(f'Invalid audio file: {input_file}')
                    continue
                
                # Process based on method
                if self.is_ensemble:
                    self.process_ensemble(input_file)
                else:
                    self.process_single(input_file)
                
                logger.info(f'Completed processing: {os.path.basename(input_file)}')
                
            except Exception as e:
                logger.error(f'Error processing {input_file}: {e}')
                import traceback
                traceback.print_exc()
            finally:
                # Clear GPU cache
                clear_gpu_cache()
                # Check output directory contents
                if os.path.exists(self.export_path):
                    logger.info(f'Output directory contents after processing: {os.listdir(self.export_path)}')
        
        return True
    
    def process_single(self, input_file):
        """Process single file with selected method"""
        # Create model data
        model_data = ModelData(
            model_name=self.model_name or self.get_default_model(),
            selected_process_method=self.process_method,
            config=self.config
        )
        
        if not model_data.model_status:
            logger.error(f'Invalid model: {model_data.model_name}')
            return
        
        # Prepare process data
        process_data = {
            'model_data': model_data,
            'export_path': self.export_path,
            'audio_file': input_file,
            'audio_file_base': os.path.splitext(os.path.basename(input_file))[0],
            'set_progress_bar': lambda step, inference_iterations=0: None,
            'write_to_console': lambda text, **kwargs: logger.info(text),
            'process_iteration': 1,
            'cached_source_callback': lambda *args, **kwargs: (None, None),
            'cached_model_source_holder': {},
            'list_all_models': [],
            'is_ensemble_master': False,
            'is_4_stem_ensemble': False
        }
        
        # Debug: Print export path
        logger.info(f'DEBUG: export_path = {self.export_path}')
        logger.info(f'DEBUG: audio_file_base = {os.path.splitext(os.path.basename(input_file))[0]}')
        logger.info(f'DEBUG: os.path.exists(export_path) = {os.path.exists(self.export_path)}')
        
        # Create separator
        if model_data.process_method == 'vr':
            separator = SeperateVR(model_data, process_data)
        elif model_data.process_method == 'mdx':
            if model_data.is_mdx_c:
                separator = SeperateMDXC(model_data, process_data)
            else:
                separator = SeperateMDX(model_data, process_data)
        elif model_data.process_method == 'demucs':
            separator = SeperateDemucs(model_data, process_data)
        else:
            logger.error(f'Unknown process method: {model_data.process_method}')
            return
        
        # Run separation
        separator.seperate()
    
    def process_ensemble(self, input_file):
        """Process file with ensemble"""
        # Create ensemble
        ensemble = Ensembler(config=self.config)
        
        # Process each model
        for model_name in self.ensemble_models:
            model_data = ModelData(
                model_name=model_name,
                selected_process_method=self.process_method,
                config=self.config
            )
            
            if not model_data.model_status:
                logger.error(f'Invalid model: {model_name}')
                continue
            
            # Prepare process data
            process_data = {
                'model_data': model_data,
                'export_path': ensemble.ensemble_folder_name,
                'audio_file': input_file,
                'audio_file_base': os.path.splitext(os.path.basename(input_file))[0],
                'set_progress_bar': lambda step, inference_iterations=0: None,
                'write_to_console': lambda text: logger.info(text),
                'process_iteration': 1,
                'cached_source_callback': lambda *args: None,
                'cached_model_source_holder': {},
                'list_all_models': [],
                'is_ensemble_master': True,
                'is_4_stem_ensemble': False
            }
            
            # Create separator
            if model_data.process_method == 'vr':
                separator = SeperateVR(model_data, process_data)
            elif model_data.process_method == 'mdx':
                if model_data.is_mdx_c:
                    separator = SeperateMDXC(model_data, process_data)
                else:
                    separator = SeperateMDX(model_data, process_data)
            elif model_data.process_method == 'demucs':
                separator = SeperateDemucs(model_data, process_data)
            else:
                logger.error(f'Unknown process method: {model_data.process_method}')
                continue
            
            # Run separation
            separator.seperate()
        
        # Ensemble outputs
        audio_file_base = os.path.splitext(os.path.basename(input_file))[0]
        
        # Determine stems to ensemble
        if self.process_method == 'demucs' and self.config.get('demucs_stems') == 'all':
            # Ensemble all stems
            from core.utils import extract_stems
            stems = extract_stems(audio_file_base, ensemble.ensemble_folder_name)
            for stem in stems:
                ensemble.ensemble_outputs(audio_file_base, ensemble.ensemble_folder_name, stem, is_4_stem=True)
        else:
            # Ensemble primary and secondary stems
            if not self.config.get('is_primary_stem_only'):
                ensemble.ensemble_outputs(audio_file_base, ensemble.ensemble_folder_name, 'Primary Stem')
            if not self.config.get('is_secondary_stem_only'):
                ensemble.ensemble_outputs(audio_file_base, ensemble.ensemble_folder_name, 'Secondary Stem')
                ensemble.ensemble_outputs(audio_file_base, ensemble.ensemble_folder_name, 'Secondary Stem', is_inst_mix=True)
    
    def get_default_model(self):
        """Get default model for selected method"""
        if self.process_method == 'vr':
            return self.config.get('vr_model', 'UVR-Model-5')
        elif self.process_method == 'mdx':
            return self.config.get('mdx_net_model', 'UVR-MDX-NET-Vocal_1')
        elif self.process_method == 'demucs':
            return self.config.get('demucs_model', 'htdemucs')
        else:
            return 'Choose Model'
    
    def run(self):
        """Run CLI"""
        try:
            # Parse arguments
            args = self.parse_args()
            
            # Configure
            self.configure(args)
            
            # Process
            success = self.process()
            
            if success:
                logger.info('Processing completed successfully')
                return 0
            else:
                logger.error('Processing failed')
                return 1
                
        except KeyboardInterrupt:
            logger.info('Processing interrupted by user')
            return 1
        except Exception as e:
            logger.error(f'Unexpected error: {e}')
            import traceback
            traceback.print_exc()
            return 1

if __name__ == '__main__':
    cli = CLI()
    sys.exit(cli.run())
