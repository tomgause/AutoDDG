import logging as log
import random
from pathlib import Path

from datumaro.components.dataset import Dataset
from datumaro.components.visualizer import Visualizer


def visualize_random_samples(dataset_path, num_samples=9, save_dir=None):
    """
    Randomly visualize a specified number of items from a Datumaro dataset using Datumaro's Visualizer.

    Args:
        dataset_path (str): Path to the Datumaro dataset directory.
        num_samples (int): Number of random items to visualize.
        save_dir (str, optional): Directory to save visualizations. If None, display interactively.
    """
    # Set up logging
    logger = log.getLogger(__name__)
    log.basicConfig(level=log.INFO)

    # Load the dataset
    dataset = Dataset.load(dataset_path)
    dataset.init_cache()

    # Ensure num_samples doesn't exceed dataset size
    num_samples = min(num_samples, len(dataset))

    # Randomly sample items from the dataset
    sampled_items = random.sample(list(dataset), num_samples)
    logger.info(f"Randomly selected {num_samples} items from the dataset.")

    # Determine grid size and scale figure size
    grid_side = int(num_samples**0.5) + 1  # Infer grid dimensions
    figsize = (grid_side * 4, grid_side * 3)  # Adjust scaling as needed

    # Initialize the Visualizer with scaled figure size
    visualizer = Visualizer(dataset, figsize=figsize, alpha=0.7)

    # Use the vis_gallery method to visualize the random items
    fig = visualizer.vis_gallery(sampled_items, grid_size=(None, None))

    if save_dir:
        # Save the visualization to the specified directory
        save_dir_path = Path(save_dir)
        save_dir_path.mkdir(parents=True, exist_ok=True)
        save_path = save_dir_path / "random_samples_gallery.png"
        fig.savefig(save_path)
        logger.info(f"Saved gallery visualization to {save_path}")
    else:
        # Display the gallery interactively
        fig.show()
