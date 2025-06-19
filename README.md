# VideoMAE-FAU-Emotion-analysis


# Data loader
📁 Folder and CSV Expectations
Each dataset is assumed to be structured like:

data/
├── videos/
│   ├── subject1_video1/
│   │   ├── 00001.jpg
│   │   ├── 00002.jpg
│   │   └── ...
│   ├── subject2_video2/
│   └── ...
├── labels.csv
And labels.csv (or split_csv) should have the format:

video_id	start_frame	AU1	AU2	... / Emotion0	Emotion1	...
subject1_video1	5	0	1	...		
subject2_video2	3	1	0	...		

🔍 Example Usage

from data.datasets import get_dataset

dataset = get_dataset(
    name="Aff-Wild2",
    split_csv="./data/affwild2_train.csv",
    root_dir="./data/videos/",
    clip_length=16,
    label_type="emotion"
)

clip, label = dataset[0]
print("Clip shape:", clip.shape)   # (T, C, H, W)
print("Label shape:", label.shape) # (num_emotions,)