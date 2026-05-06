import os
import zipfile
import json
from tqdm import tqdm

dataset_dir = "./datasets/video-mme"
zip_files = [f for f in os.listdir(dataset_dir) if f.startswith("videos_chunked_") and f.endswith(".zip")]
zip_files.sort()

video_map = {}

print(f"Found {len(zip_files)} zip files.")

for zf in tqdm(zip_files):
    zip_path = os.path.join(dataset_dir, zf)
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            for name in z.namelist():
                if name.endswith(".mp4") or name.endswith(".mkv") or name.endswith(".avi"):
                    # name is like "data/videoID.mp4"
                    filename = os.path.basename(name)
                    video_id = os.path.splitext(filename)[0]
                    video_map[video_id] = {"zip_file": zf, "rel_path": name}
    except Exception as e:
        print(f"Error reading {zf}: {e}")

output_path = "./eval/video_map.json"
with open(output_path, "w") as f:
    json.dump(video_map, f, indent=2)

print(f"Mapped {len(video_map)} videos. Saved to {output_path}")
