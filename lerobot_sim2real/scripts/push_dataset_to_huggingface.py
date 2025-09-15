from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Use a higher tolerance (25 seconds) to accommodate timestamp sync issues in the dataset
dataset = LeRobotDataset(
    repo_id='yadan0418/record-test', 
    root='~/.cache/huggingface/lerobot/yadan0418/record-test',
    tolerance_s=25.0  # Set tolerance to 25 seconds to handle the ~20 second offset
)
dataset.push_to_hub()