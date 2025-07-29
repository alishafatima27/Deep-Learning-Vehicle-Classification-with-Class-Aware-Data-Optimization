import os
import shutil
import json
from datetime import datetime
from pathlib import Path
from cleanvision import Imagelab
from tqdm import tqdm
from PIL import Image

class DataCleaner:
    def __init__(self, config):
        self.train_dir = Path(config['train_dir'])
        self.val_dir = Path(config['val_dir'])
        self.output_dir = Path(config['output_dir'])
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.cleaning_log = {
            'timestamp': self.timestamp,
            'initial_stats': {},
            'cleaning_actions': [],
            'final_stats': {},
            'removed_files': {'corrupt': [], 'duplicates': [], 'low_quality': []}
        }
        self.create_output_directories()

    def create_output_directories(self):
        directories = [
            'reports', 'removed_files/corrupt', 'removed_files/duplicates',
            'removed_files/low_quality', 'logs', 'cleaned_dataset/train', 'cleaned_dataset/val'
        ]
        for dir_name in directories:
            (self.output_dir / dir_name).mkdir(parents=True, exist_ok=True)

    def log_action(self, action, details):
        self.cleaning_log['cleaning_actions'].append({
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'details': details
        })

    def collect_stats(self, directory):
        stats, total_files = {}, 0
        for class_dir in Path(directory).iterdir():
            if class_dir.is_dir():
                file_count = len(list(class_dir.glob('*')))
                stats[class_dir.name] = file_count
                total_files += file_count
        return stats, total_files

    def check_corrupt_image(self, file_path):
        try:
            with Image.open(file_path) as img:
                img.verify()
            return False
        except:
            return True

    def run_cleanvision(self, data_path, split):
        try:
            imagelab = Imagelab(data_path=str(data_path))
            imagelab.find_issues(issue_types={
                'dark': {}, 'light': {}, 'odd_aspect_ratio': {},
                'low_information': {}, 'blurry': {}, 'exact_duplicates': {}, 'near_duplicates': {}
            })
            save_path = self.output_dir / 'reports' / f'cleanvision_{split}_results'
            imagelab.save(str(save_path))
            self.log_action(f"CleanVision {split} Analysis", f"Saved to {save_path}")
            return imagelab
        except Exception as e:
            self.log_action(f"CleanVision {split} Failed", str(e))
            return None

    def execute_cleaning(self):
        files_to_remove = {'corrupt': set(), 'duplicates': set(), 'low_quality': set()}
        cleaned_files = {'train': set(), 'val': set()}

        # Collect all images
        for split, dir_path in [('train', self.train_dir), ('val', self.val_dir)]:
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            image_paths = []
            for root, _, files in os.walk(dir_path):
                for file in files:
                    if Path(file).suffix.lower() in image_extensions:
                        image_paths.append(os.path.join(root, file))
            cleaned_files[split] = set(image_paths)
            print(f"Found {len(image_paths)} images in {split} directory")

            # Check for corrupt images
            for file_path in image_paths:
                if self.check_corrupt_image(file_path):
                    files_to_remove['corrupt'].add(file_path)
                    self.log_action("Corrupt file detected", str(file_path))

        # Run CleanVision
        train_imagelab = self.run_cleanvision(self.train_dir, 'train')
        val_imagelab = self.run_cleanvision(self.val_dir, 'val')

        # Process CleanVision issues
        for split, imagelab in [('train', train_imagelab), ('val', val_imagelab)]:
            if imagelab and hasattr(imagelab, 'issues'):
                issue_types = ['dark', 'light', 'odd_aspect_ratio', 'low_information', 'blurry']
                for issue in issue_types:
                    bool_col = f'is_{issue}_issue'
                    if bool_col in imagelab.issues.columns:
                        issue_files = imagelab.issues[imagelab.issues[bool_col]].index.tolist()
                        files_to_remove['low_quality'].update(issue_files)
                        self.log_action(f"{split} {issue} issues", f"Found {len(issue_files)} files")

                # Handle duplicates
                for dup_type in ['exact_duplicates', 'near_duplicates']:
                    if dup_type in imagelab.info and 'sets' in imagelab.info[dup_type]:
                        for dup_set in imagelab.info[dup_type]['sets']:
                            if len(dup_set) > 1:
                                files_to_remove['duplicates'].update(dup_set[1:])
                                self.log_action(f"{split} {dup_type}", f"Kept {dup_set[0]}, removed {len(dup_set[1:])}")

        # Update cleaned files
        for split in ['train', 'val']:
            cleaned_files[split] = cleaned_files[split] - set().union(*files_to_remove.values())

        # Copy cleaned files
        for split, src_dir in [('train', self.train_dir), ('val', self.val_dir)]:
            print(f"Copying cleaned {split} files...")
            for file_path in tqdm(cleaned_files[split], desc=f"Copying {split} files"):
                try:
                    relative_path = Path(file_path).relative_to(src_dir)
                    new_path = self.output_dir / 'cleaned_dataset' / split / relative_path
                    new_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(file_path, new_path)
                    self.log_action(f"Copied clean {split} file", f"{file_path} to {new_path}")
                except Exception as e:
                    self.log_action(f"Copy failed {split}", f"{file_path}: {str(e)}")

        # Save logs and stats
        self.cleaning_log['initial_stats'] = {
            'train': self.collect_stats(self.train_dir),
            'val': self.collect_stats(self.val_dir)
        }
        self.cleaning_log['final_stats'] = {
            'train': self.collect_stats(self.output_dir / 'cleaned_dataset/train'),
            'val': self.collect_stats(self.output_dir / 'cleaned_dataset/val')
        }
        self.cleaning_log['removed_files'] = {k: list(v) for k, v in files_to_remove.items()}
        log_path = self.output_dir / 'logs' / f'cleaning_complete_{self.timestamp}.json'
        with open(log_path, 'w') as f:
            json.dump(self.cleaning_log, f, indent=2)
        print(f"Cleaning log saved to: {log_path}")

def main(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    cleaner = DataCleaner(config)
    cleaner.execute_cleaning()
    print("🎉 Data cleaning complete!")

if __name__ == "__main__":
    config_path = r"C:\Users\HP\Desktop\vehicle_classification\config.json"
    main(config_path)