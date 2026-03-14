# Professional Singer Separator - Enhanced Version 🎤✨

## 🆕 New Features

This enhanced version adds powerful visualization and interaction features:

### ✅ Model Pre-download
- **One-click model download** before processing
- Pre-cache Pyannote and Resemblyzer models
- Avoid delays during first-time processing

### ✅ Interactive Threshold Slider
- **Real-time threshold adjustment** (0.50 - 0.90)
- Visual feedback showing current value
- Fine-tune sensitivity on the fly

### ✅ Waveform & Spectrogram Visualization
- **Dual-view display**: waveform + spectrogram
- **Color-coded speaker segments** overlaid on both views
- Interactive matplotlib canvas with zoom/pan
- Automatic legend showing all detected speakers

### ✅ Audio Playback Controls
- **Play/pause** processed audio directly in app
- **Volume control** slider
- **Position indicator** (MM:SS / MM:SS)
- **Stop button** to reset playback

### ✅ Visual Segment Annotations
- Each speaker assigned a unique color
- Segments highlighted on waveform (transparent overlay)
- Segments outlined on spectrogram (colored boxes)
- Easy visual identification of speaker distribution

---

## 📦 Installation

### Step 1: Install Dependencies

```bash
pip install -r requirements_enhanced.txt
```

Or manually:
```bash
pip install torch librosa soundfile scikit-learn scipy PyQt6 PyQt6-Multimedia matplotlib
pip install pyannote.audio resemblyzer
```

### Step 2: Get HuggingFace Token

For best quality (Pyannote), you need a HuggingFace token:

1. Create account: https://huggingface.co/join
2. Get token: https://huggingface.co/settings/tokens
3. Accept model agreements:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0

4. Set token:
   ```bash
   export HF_TOKEN="hf_your_token_here"
   ```

### Step 3: Run

```bash
python pro_singer_separator_enhanced.py
```

---

## 🎯 Quick Start Guide

### 1. Pre-download Models (Recommended)

Before processing any audio:
1. Click **"📥 Pre-download Models"**
2. Enter your HuggingFace token (if using Pyannote)
3. Wait for download to complete
4. Models are now cached locally

### 2. Process Audio

1. **Add Files**: Click "Add Files..." and select vocal tracks
2. **Select Output**: Choose output directory
3. **Optional Reference**: Add 3-10 sec sample of target singer
4. **Adjust Threshold**: Use slider to set similarity threshold
   - **0.60-0.65**: More inclusive (catches more)
   - **0.70**: Balanced (recommended)
   - **0.75-0.80**: More precise (stricter)
5. **Configure Settings**:
   - Speaker selection (if no reference)
   - Number of speakers (auto-detect or specify)
6. Click **"▶ Start Processing"**

### 3. Visualize Results

After processing completes:
1. Click **"👁 Visualize Last Processed File"**
2. **Visualization window opens** showing:
   - Top: Waveform with color-coded speaker segments
   - Bottom: Spectrogram with segment overlays
   - Legend: Speaker identification
3. **Use playback controls**:
   - ▶/⏸ Play/Pause
   - ⏹ Stop
   - 🔊 Volume slider
4. **Navigate visualization**:
   - Zoom: Scroll or toolbar zoom tool
   - Pan: Click and drag or toolbar pan tool
   - Reset: Home button in toolbar

---

## 🎨 Visualization Features

### Waveform Display
- **Blue line**: Audio amplitude over time
- **Colored overlays**: Speaker segments (transparent)
- **Grid**: Time and amplitude reference
- **Legend**: Speaker identification (top right)

### Spectrogram Display
- **Color gradient**: Frequency content (viridis colormap)
- **Y-axis**: Frequency (kHz)
- **X-axis**: Time (seconds)
- **Colored boxes**: Speaker segments
- **Colorbar**: dB scale reference

### Speaker Colors
- **Speaker 1**: Red (#FF6B6B)
- **Speaker 2**: Teal (#4ECDC4)
- **Speaker 3**: Blue (#45B7D1)
- **Speaker 4**: Coral (#FFA07A)
- **Speaker 5**: Mint (#98D8C8)

---

## 🎵 Audio Playback

### Controls
- **▶ Play**: Start playback from current position
- **⏸ Pause**: Pause at current position
- **⏹ Stop**: Stop and reset to beginning
- **Volume**: Adjust from 0-100%
- **Position**: Shows current time / total duration

### Supported Formats
- WAV, MP3, FLAC, M4A

### Notes
- Playback uses original input file (not processed output)
- Visualization shows where each speaker was detected
- Use this to verify segmentation quality

---

## ⚙️ Advanced Settings

### Similarity Threshold Slider

**What it does**: Controls how similar a segment must be to the reference

**Slider range**: 0.50 - 0.90

**Visual feedback**: Shows current value (e.g., "0.70")

**When to adjust**:
- **Lower (0.60-0.65)**: 
  - Target singer has varied styles
  - Recording has different dynamics
  - Want to capture everything
  
- **Higher (0.75-0.80)**:
  - Other singers sound very similar
  - Want maximum precision
  - Better to miss some than include wrong voice

**Live adjustment**: Change anytime before processing

### Speaker Selection

When **not** using reference audio:
- **Speaker 1 (Most speech)**: Usually the lead singer
- **Speaker 2**: Often featured artist or harmonies
- **Speaker 3**: Typically backup vocals

Based on **total duration** in the track

### Number of Speakers

- **Auto-detect**: Let the model decide (recommended)
- **2-5**: Specify exact number
  - Better for very short tracks
  - Improves consistency in batch processing
  - Use if you know the exact count

---

## 🔍 Interpreting Visualizations

### Good Segmentation
✅ Clear color separation  
✅ Segments align with actual singer changes  
✅ No tiny fragments (< 0.5s)  
✅ Smooth transitions  
✅ Target singer fully covered  

### Poor Segmentation
❌ Mixed colors in single-speaker sections  
❌ Many tiny segments (flickering)  
❌ Missing obvious target segments  
❌ Incorrect speaker assignment  

**Solutions**:
- Adjust similarity threshold
- Try different speaker selection
- Provide better reference sample
- Check input audio quality

---

## 💡 Workflow Tips

### Best Practice Workflow

1. **First Run**: No reference
   - Use auto-detection
   - Visualize results
   - Identify which speaker is your target

2. **Second Run**: With reference (if available)
   - Add 5-10 sec clean sample
   - Start with threshold 0.70
   - Visualize and check coverage

3. **Fine-tune**: Adjust threshold
   - Too much retained? Increase threshold
   - Too little retained? Decrease threshold
   - Check visualization after each adjustment

### Quality Checklist

Before processing a batch:
- ✓ Input vocals are clean (no music bleed)
- ✓ Reference sample is high quality
- ✓ Reference matches target singer
- ✓ Settings configured appropriately
- ✓ Tested on one file first

---

## 🎯 Example Use Cases

### Case 1: Duet with Reference
```
Input: Female duet (soprano + alto)
Reference: 10 sec solo from soprano
Threshold: 0.70
Result: Clean soprano extraction
Visualization: Red (soprano) and teal (alto) clearly separated
```

### Case 2: Group Song, No Reference
```
Input: 3-person group vocal
Settings: Auto-detect, Speaker 1
Result: Lead singer extracted
Visualization: Three colors showing all singers
Note: Verified Speaker 1 was correct via playback
```

### Case 3: Dynamic Performance
```
Input: Singer doing soft verse + powerful chorus
Reference: Mixed dynamics sample
Threshold: 0.65 (lower for variation)
Result: Complete singer extracted across dynamics
Visualization: Single color throughout, no gaps
```

---

## 🐛 Troubleshooting

### "No audio retained" / Empty output

**Possible causes**:
1. Reference doesn't match input
2. Threshold too high
3. Wrong speaker selected

**Solutions**:
- ✓ Visualize input to verify speakers
- ✓ Lower threshold to 0.60-0.65
- ✓ Try non-reference mode first
- ✓ Check reference quality

### Flickering / Fragmented segments

**Causes**: Inconsistent similarity scores

**Solutions**:
- ✓ Increase threshold slightly
- ✓ Use longer/better reference sample
- ✓ Temporal smoothing helps (automatic)

### Visualization window won't open

**Causes**: Missing matplotlib or Qt multimedia

**Solutions**:
```bash
pip install --upgrade matplotlib PyQt6-Multimedia
```

### Playback not working

**Causes**: Qt multimedia not installed or codec missing

**Solutions**:
```bash
pip install PyQt6-Multimedia
```

On Linux, install codecs:
```bash
sudo apt install libqt6multimedia6-plugins
```

### Model download fails

**Causes**: 
- No HuggingFace token
- Token not accepted model agreements
- Network issues

**Solutions**:
- ✓ Verify token is correct
- ✓ Accept all model agreements
- ✓ Check internet connection
- ✓ Use Resemblyzer fallback (no token needed)

---

## 📊 Performance

### Processing Speed

| Hardware | Speed (Pyannote) | Speed (Resemblyzer) |
|----------|------------------|---------------------|
| CPU (Intel i7) | ~3x real-time | ~5x real-time |
| GPU (RTX 3080) | ~20x real-time | ~30x real-time |
| Apple M1 (MPS) | ~10x real-time | ~15x real-time |

### Memory Usage

- **Pyannote**: ~2-4 GB RAM
- **Resemblyzer**: ~1-2 GB RAM
- **Visualization**: +500 MB per audio file

### Model Sizes

- **Pyannote models**: ~100 MB total
- **Resemblyzer model**: ~20 MB
- **Cached locally** after first download

---

## 🔧 Technical Details

### Visualization Implementation

**Waveform**:
- Matplotlib line plot
- Rectangle patches for segment overlays
- Automatic y-axis scaling

**Spectrogram**:
- Librosa STFT (n_fft=2048, hop=512)
- dB scale normalization
- Viridis colormap
- Rectangle patches for segments

**Playback**:
- Qt6 QMediaPlayer
- QAudioOutput for volume control
- 100ms position update interval

### Threshold Slider

**Implementation**:
- QSlider (Horizontal)
- Range: 50-90 (represents 0.50-0.90)
- Tick marks every 5 units
- Real-time label update

### Model Pre-download

**Process**:
1. QThread for background download
2. Progress signals to GUI
3. Downloads to HuggingFace cache (~/.cache/huggingface)
4. Validates successful download

---

## 🆚 Comparison: Standard vs Enhanced

| Feature | Standard | Enhanced |
|---------|----------|----------|
| Model download | Auto on first use | ✅ Pre-download button |
| Threshold | Fixed dropdown | ✅ Interactive slider |
| Visualization | None | ✅ Waveform + Spectrogram |
| Segment view | Text log only | ✅ Color-coded overlays |
| Playback | None | ✅ Built-in player |
| Speaker colors | N/A | ✅ 5 distinct colors |
| Verification | Output audio only | ✅ Visual + audio |

---

## 📝 Keyboard Shortcuts (Visualization)

- **Space**: Play/Pause
- **Home**: Reset zoom
- **Arrow keys**: Pan view
- **Scroll**: Zoom in/out
- **Escape**: Close visualization

---

## 🎓 Learning Resources

### Understanding the Visualization

**Waveform**: Shows amplitude (loudness) over time
- Peaks = loud moments
- Flat = quiet/silence
- Colored overlays = detected speakers

**Spectrogram**: Shows frequency content over time
- Bright = strong frequency presence
- Dark = weak/absent frequencies
- Horizontal lines = sustained tones
- Vertical patterns = transients/attacks

**Speaker Segments**: Continuous blocks of same color
- Each color = one speaker
- Boundaries = speaker changes
- Gaps = unassigned (usually silence)

---

## 🤝 Contributing

Suggestions for improvement? Found a bug?

Areas for contribution:
- Additional visualization options
- Export segment timestamps
- Batch visualization
- Real-time threshold adjustment
- Custom color schemes

---

## 📄 License

Same as state-of-the-art version:
- Pyannote-audio: MIT License
- Resemblyzer: MIT License
- PyQt6: GPL/Commercial
- Matplotlib: PSF License

---

## 🙏 Acknowledgments

Enhanced version builds on:
- Pyannote-audio (Hervé Bredin et al.)
- Resemblyzer (Corentin Jemine)
- Matplotlib (John Hunter et al.)
- PyQt6 (Riverbank Computing)

---

## 🚀 What's Next?

Planned features for future versions:
- [ ] Real-time waveform annotation during playback
- [ ] Export segment timestamps to JSON/CSV
- [ ] Batch visualization mode
- [ ] Compare multiple threshold values
- [ ] Custom speaker color selection
- [ ] Segment editing (manual adjustment)
- [ ] Export visualization as image

---

## 💡 Pro Tips

1. **Pre-download before important work** - avoid delays
2. **Always visualize first file** - verify settings work
3. **Use playback to verify** - listen to segment quality
4. **Save threshold settings** - note what works best
5. **Batch similar files** - use same settings for consistency
6. **Check spectrogram** - catches errors waveform might hide
7. **Lower threshold for varied singing** - captures dynamics
8. **Higher threshold for similar voices** - increases precision

Enjoy the enhanced singer separation experience! 🎵
