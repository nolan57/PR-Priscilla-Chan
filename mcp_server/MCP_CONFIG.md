# MCP Server Configuration Examples

## For OpenCode

Add to your OpenCode configuration:

```json
{
  "mcpServers": {
    "pcs-audio": {
      "command": "uv",
      "args": ["--directory", "/Users/lpcw/Documents/PCS", "run", "mcp_server"],
      "env": {
        "PYTHONPATH": "/Users/lpcw/Documents/PCS"
      }
    }
  }
}
```

## Available Tools

### Audio Processing Tools

| Tool              | Description                                 |
| ----------------- | ------------------------------------------- |
| `separate_vocals` | Separate vocals from mixed audio using UVR5 |
| `mute_audio`      | Mute specified regions in audio             |
| `trim_audio`      | Trim audio to specified time range          |
| `remove_harmony`  | Remove harmony using spectral masking       |
| `merge_audio`     | Merge multiple audio files                  |

### Pipeline Tools

| Tool                   | Description                   |
| ---------------------- | ----------------------------- |
| `extract_f0`           | Extract high-precision F0     |
| `build_diffsinger_npz` | Build DiffSinger training NPZ |

### Workflow Tools (Skills)

| Tool                      | Description                        |
| ------------------------- | ---------------------------------- |
| `run_vocoder_workflow`    | Full vocoder data prep workflow    |
| `run_diffsinger_workflow` | Full DiffSinger data prep workflow |
| `get_workflow_status`     | Get workflow processing status     |

## Example AI Commands

> "Use run_vocoder_workflow to process ./raw_songs and output to ./data"

> "Extract F0 from audio.wav using extract_f0 tool"

> "Check workflow status using get_workflow_status"
