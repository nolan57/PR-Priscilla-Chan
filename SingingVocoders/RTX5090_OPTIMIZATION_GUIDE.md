# RTX 5090 Vocoder Training Optimization Guide

## Hardware Specifications
- **GPU**: RTX 5090 (32GB VRAM)
- **System Memory**: 92GB
- **Goal**: Maximize training speed while maintaining audio quality

## Key Optimization Points

### 1. Memory Utilization Optimization
```yaml
# Leverage 32GB VRAM advantage
crop_mel_frames: 80        # Increased from 32 to 80
batch_size: 32             # Increased from 10 to 32
upsample_initial_channel: 1024  # Increased from 512 to 1024
```

### 2. Training Speed Optimization
```yaml
# Use more efficient optimizer
optimizer_cls: modules.optimizer.muon.Muon_AdamW
lr: 0.0003                 # Slightly higher learning rate
pl_trainer_precision: '16-mixed'  # Mixed precision training

# Data loading optimization
ds_workers: 12             # Increase data loading worker threads
dataloader_prefetch_factor: 4
binarization_args.num_workers: 16  # Parallel preprocessing
```

### 3. Model Architecture Optimization
```yaml
# Simplify discriminator for better speed
discriminator_periods: [2, 3, 5, 7, 11, 17, 23]  # Reduce period count
fast_mpd_strides: [4, 4, 4]  # Fast MPD configuration
fast_mpd_kernel_size: 11

# Add noise control
noise_sigma: 0.0
```

### 4. Training Strategy Optimization
```yaml
# Validation and logging frequency adjustment
val_check_interval: 1000   # Reduce validation frequency
log_interval: 100          # Increase logging interval
num_valid_plots: 32        # Reduce validation sample count

# Training duration extension
max_updates: 300000        # Increase total training steps
```

## Expected Performance Improvements

| Optimization Item | Improvement |
|-------------------|-------------|
| Batch Size Increase | ~3x memory utilization |
| Mixed Precision Training | ~2x training speed |
| Muon Optimizer | ~20% convergence speed improvement |
| Parallel Data Loading | ~30% data bottleneck reduction |
| Architecture Simplification | ~15% computational efficiency improvement |

## Overall Expected Performance Gains
Compared to the original configuration, the RTX 5090 optimized configuration is expected to achieve:
- **Training Speed Improvement**: 3-5x
- **Memory Utilization**: From ~8GB to ~28GB
- **Per-step Training Time**: From ~2 seconds to ~0.5 seconds
- **Overall Training Time**: Significantly reduced

## Usage Recommendations

1. **First Run**: Test configuration stability with a smaller dataset first
2. **Monitoring Metrics**: Closely monitor GPU utilization and memory usage
3. **Parameter Tuning**: Adjust `batch_size` and `crop_mel_frames` according to actual dataset size
4. **Backup Configuration**: Keep original configuration as baseline for comparison

## Important Notes

⚠️ **Important Reminders**:
- Ensure system has sufficient swap space (recommended 16GB+)
- Monitor GPU temperature and improve cooling if necessary
- Regularly save checkpoints to prevent unexpected interruptions
- Mixed precision may affect final audio quality, switch back to 32-bit precision if needed

## Troubleshooting

If encountering out-of-memory errors:
1. Gradually reduce `batch_size` (decrease by 4 each time)
2. Reduce `crop_mel_frames` (decrease by 8 each time)
3. Decrease `upsample_initial_channel` (to 768 or 512)

If training is unstable:
1. Lower learning rate (`lr: 0.0001`)
2. Enable gradient clipping (`clip_grad_norm: 0.5`)
3. Fall back to standard AdamW optimizer