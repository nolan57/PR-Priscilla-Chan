# Fine-tuning Parameter Guide for Singer Reproduction

## Goal
Perfectly reproduce a specific singer's timbre, vocal techniques, and style to perform any song.

## Key Components to Preserve (Frozen Parameters)

### 1. Pitch Generation Core
```yaml
"generator.m_source"           # Core NSF pitch source module
"generator.l_sin_gen"          # Sine wave generator for fundamental frequency
"generator.l_linear"           # Linear layer for harmonic merging  
"generator.l_tanh"             # Tanh activation for source waveform shaping
```
**Reason**: These components define the fundamental pitch generation and harmonic structure that characterize the singer's unique vocal production.

### 2. Timbre Foundation Layers
```yaml
"generator.conv_pre"           # Initial mel-spectrogram processing
"generator.ups.0"              # First upsampling (establishes core timbre)
"generator.ups.1"              # Second upsampling (important base characteristics)
```
**Reason**: Early layers capture the singer's fundamental tonal qualities and spectral envelope characteristics.

### 3. Harmonic Processing Blocks
```yaml
"generator.resblocks.0"        # First residual blocks (core harmonic processing)
"generator.resblocks.1"        # Second residual blocks (fundamental tone shaping)
"generator.resblocks.2"        # Third residual blocks (basic timbre formation)
```
**Reason**: Early residual blocks handle the core harmonic content that defines vocal timbre.

### 4. Vocal Texture Elements
```yaml
"generator.noise_convs"        # Noise injection for breathiness and vocal texture
"generator.source_conv"        # Source harmonic injection mechanisms
```
**Reason**: These preserve the singer's characteristic vocal textures, breath control, and subtle noise components.

## Parameters Allowed to Adapt

### Style and Expression Layers
- Later upsampling layers (`generator.ups.2`, `generator.ups.3`, `generator.ups.4`)
- Later residual blocks (`generator.resblocks.3+`)
- Final output processing (`generator.conv_post`)

### Why This Approach Works

1. **Preserves Identity**: Keeps the core vocal fingerprint intact
2. **Enables Style Transfer**: Allows adaptation to new songs while maintaining singer characteristics
3. **Prevents Catastrophic Forgetting**: Maintains pretrained knowledge of the singer's voice
4. **Optimizes Learning**: Focuses adaptation on expressive elements rather than fundamental characteristics

## Usage Instructions

1. Set `finetune_enabled: true`
2. Point `finetune_ckpt_path` to your pretrained singer model
3. Use the `finetune_ignored_params` list above
4. Train on new songs/speech data from the target singer

## Expected Results

- Maintains original singer's distinctive timbre
- Adapts to new vocal styles and song characteristics
- Preserves unique vocal techniques (vibrato, breath control, etc.)
- Enables performance of arbitrary songs in the singer's style
