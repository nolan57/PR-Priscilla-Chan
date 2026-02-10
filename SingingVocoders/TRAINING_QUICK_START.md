# Vocoder Training Quick Start Guide

## Recommended Configuration Selection

### For RTX 5090 Users (32GB VRAM):
Use `configs/base_hifi_rtx5090_optimized.yaml` - Fully optimized for your hardware

### For Other High-end GPU Users:
Use `configs/base_hifi_gpu.yaml` - Balanced performance and stability

### For Standard Configurations:
Use `configs/base_hifi_cpu.yaml` - CPU training configuration

## Training Command Examples

```bash
# Using RTX 5090 optimized configuration
python train.py --config configs/base_hifi_rtx5090_optimized.yaml --exp_name my_vocoder_exp

# Using standard GPU configuration  
python train.py --config configs/base_hifi_gpu.yaml --exp_name my_vocoder_exp

# Specify working directory
python train.py --config configs/base_hifi_rtx5090_optimized.yaml --exp_name my_experiment --work_dir ./my_experiments
```

## Key Parameter Explanations

### Memory-related Parameters
- `batch_size`: Number of samples processed per batch (32 recommended for RTX 5090)
- `crop_mel_frames`: Time frames per sample (affects memory usage)
- `upsample_initial_channel`: Model initial channel count (affects model size)

### Performance-related Parameters
- `pl_trainer_precision`: Training precision (16-mixed faster, 32-true more stable)
- `ds_workers`: Data loading worker thread count
- `dataloader_prefetch_factor`: Data prefetch factor

### Optimizer Parameters
- `optimizer_cls`: Optimizer type (Muon optimizer trains faster)
- `lr`: Learning rate (affects convergence speed)

## Monitoring Training Status

The following outputs are generated during training:
- **TensorBoard logs**: Located in `experiments/exp_name/lightning_logs/` directory
- **Checkpoint files**: Periodically saved model weights
- **Terminal output**: Real-time display of loss values and training progress

Start TensorBoard to view training curves:
```bash
tensorboard --logdir experiments/my_vocoder_exp/lightning_logs/
```

## Common Problem Solutions

### Out of Memory
- Reduce `batch_size`
- Decrease `crop_mel_frames`
- Reduce `upsample_initial_channel`

### Training Instability
- Lower learning rate
- Enable gradient clipping
- Use more conservative optimizer

### Training Too Slow
- Enable mixed precision training
- Increase data loading worker threads
- Use faster optimizer

## Performance Benchmark Reference

Expected performance on RTX 5090:
- Per-step training time: ~0.3-0.5 seconds
- Memory usage: ~25-28GB
- GPU utilization: ~85-95%
- Full training time: 24-48 hours (depending on dataset size)