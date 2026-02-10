# Linux/macOS: 批量转换当前目录及子目录所有 .wav 为 mono
find ./dataset/keep -type f -name "*.wav" | while read f; do
    out="./dataset/mono$(dirname "$f")"
    mkdir -p "$out"
    ffmpeg -i "$f" -ac 1 -ar 44100 "$out/$(basename "$f")" -y
done
