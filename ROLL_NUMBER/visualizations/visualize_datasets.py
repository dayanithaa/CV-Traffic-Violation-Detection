import os
import glob
import json
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

def plot_class_distributions(stats_json_path, output_path):
    if not os.path.exists(stats_json_path):
        return
    with open(stats_json_path, 'r') as f:
        stats = json.load(f)
        
    classes = stats.get('classes', {})
    if not classes:
        return
        
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(classes.keys()), y=list(classes.values()), palette='viridis')
    plt.title("Class Distribution")
    plt.xlabel("Class ID")
    plt.ylabel("Count")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved class distribution plot to {output_path}")

def plot_bbox_size_distributions(labels_dir, output_path):
    areas = []
    for txt_file in glob.glob(f"{labels_dir}/**/*.txt", recursive=True):
        with open(txt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    w, h = float(parts[3]), float(parts[4])
                    areas.append(w * h)
                    
    if areas:
        plt.figure(figsize=(10, 6))
        sns.histplot(areas, bins=50, kde=True, color='purple')
        plt.title("Bounding Box Size Distribution (Relative Area)")
        plt.xlabel("Relative Area (Width * Height)")
        plt.ylabel("Frequency")
        plt.savefig(output_path)
        plt.close()
        print(f"Saved bbox size distribution plot to {output_path}")

def visualize_difficult_samples(img_paths, output_path):
    if not img_paths:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i < len(img_paths):
            img = cv2.imread(img_paths[i])
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved difficult samples plot to {output_path}")

if __name__ == "__main__":
    print("Visualization utilities ready.")
    # plot_class_distributions("datasets/helmet_stats.json", "visualizations/helmet_class_dist.png")
    # plot_bbox_size_distributions("datasets/processed/helmet/labels", "visualizations/helmet_bbox_size_dist.png")
