# Professional Singer Voice Separator

A PyQt6-based application implementing a hybrid approach for professional singer voice isolation from mixed vocal tracks.

## Features

### Advanced Hybrid Technology
- **Demucs Integration**: State-of-the-art neural source separation for initial vocal extraction
- **Speaker Recognition**: Precise target speaker identification using SpeechBrain models
- **Multi-Reference Support**: Handle multiple reference clips for robust speaker matching
- **Frame-by-Frame Analysis**: Detailed voice separation with customizable thresholds
- **Professional UI**: Intuitive interface similar to singer_cleaner.py

### Key Improvements Over Basic Approach
1. **Better Initial Separation**: Uses Demucs instead of simple VAD for cleaner vocal extraction
2. **Advanced Speaker Matching**: Neural embeddings vs basic similarity scoring
3. **Robust Reference Handling**: Multiple reference segments for reliable identification
4. **Precise Masking**: Frame-level processing for detailed voice isolation
5. **Visual Feedback**: Dual waveform display showing before/after processing

## Installation

### Prerequisites
```bash
# Install system dependencies
brew install ffmpeg  # macOS
# or
sudo apt install ffmpeg  # Ubuntu/Debian
```

### Python Dependencies
```bash
# Basic installation (recommended for most users)
pip install -r requirements_pro_separator.txt

# Optional: Enhanced capabilities
pip install demucs  # For better initial separation (8GB+ RAM recommended)
pip install openai-whisper  # For precise timestamp detection
```

## Usage

### Quick Start
1. Run the application:
   ```bash
   python pro_singer_separator.py
   ```

2. Load your mixed vocal track:
   - Click "Load Mixed Vocals"
   - Select your audio file (wav/mp3/flac/aac)

3. Add reference segments:
   - Click "Add Reference Segment" 
   - Select clean clips of your target singer's voice
   - Add multiple references for better accuracy

4. Process the audio:
   - Adjust the similarity threshold (0.50-0.95)
   - Click "Separate Target Voice"
   - Wait for processing to complete

5. Save results:
   - Click "Save Isolated Vocal" when processing is finished

### Interface Overview

#### Left Panel (Controls)
- **Model Status**: Shows loaded AI models
- **Input Files**: Load mixed vocals and reference segments
- **Processing**: Adjust threshold and start separation
- **Output**: Save the isolated vocal track
- **Playback**: Listen to original vs processed audio

#### Right Panel (Visualization)
- **Waveform Display**: Shows both original (cyan) and isolated (red) waveforms
- **Real-time Playback**: Yellow cursor shows current playback position

### Best Practices

#### For Optimal Results:
1. **Reference Quality**: Use 2-3 clean reference clips (2-5 seconds each)
2. **Threshold Setting**: Start with 0.75, adjust based on results
3. **Audio Format**: WAV files work best for highest quality
4. **RAM Requirements**: 8GB+ recommended when using Demucs

#### Troubleshooting:
- **Slow Processing**: Reduce audio length or disable Demucs
- **Poor Separation**: Add more diverse reference segments
- **Memory Issues**: Use smaller audio files or basic separation mode
- **Demucs Not Available**: Install `pip install demucs` or `pip install julius` for local UVR integration
- **Low Quality Results**: Ensure reference clips are clean and representative of the target singer

## Technical Architecture

### Processing Pipeline
```
1. Audio Input → Demucs Separation (UVR-integrated) → Clean Vocals
2. Reference Clips → Speaker Embeddings → Target Profile
3. Clean Vocals → Adaptive Frame Analysis → Enhanced Speaker Matching
4. Matching Scores → Morphological Mask → Isolated Vocal
```

### UVR Component Integration
This application leverages proven components from the Ultimate Vocal Remover project:

- **Demucs Engine**: Uses the same Demucs implementation that powers UVR's professional separation
- **Spectral Utilities**: Integrates UVR's advanced `wave_to_spectrogram` and `spectrogram_to_wave` functions
- **Parameter Optimization**: Employs UVR's `determine_autoset_model` for adaptive processing
- **Quality Assurance**: Benefits from UVR's extensive testing and optimization

The integration maintains full compatibility with existing UVR workflows while providing specialized singer voice isolation capabilities.

### Key Components
- `HybridVoiceSeparator`: Main processing engine
- `SeparationWorker`: Background processing thread
- `WaveformDisplayWidget`: Dual-channel visualization
- PyQt6 GUI: Professional user interface

## Comparison with singer_cleaner.py

| Feature | singer_cleaner.py | pro_singer_separator.py |
|---------|------------------|------------------------|
| Voice Separation | Basic VAD + similarity | Demucs + neural embeddings |
| Speaker Matching | Cosine similarity | Multi-reference neural matching |
| Processing Quality | Good for simple cases | Professional grade |
| Memory Usage | Low (~2GB) | Higher with Demucs (~8GB) |
| Processing Speed | Fast | Slower but more accurate |
| Reference Handling | Single segments | Multiple references supported |

## License and Credits

This application builds upon:
- **Demucs**: Facebook Research (MIT License)
- **SpeechBrain**:speechbrain (Apache 2.0 License)  
- **PyQt6**: Riverbank Computing (GPL v3)
- **LibROSA**: librosa developers (ISC License)

## Support

For issues or questions:
1. Check the model status panel for loaded components
2. Ensure all dependencies are properly installed
3. Verify audio files are not corrupted
4. Try adjusting the similarity threshold

The application will automatically fall back to basic methods when advanced models aren't available.