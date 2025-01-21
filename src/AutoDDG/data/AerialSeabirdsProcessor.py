import logging
import random
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import rasterio
from datumaro.components.annotation import (
    AnnotationType,
    LabelCategories,
    Points,
)
from datumaro.components.dataset import Dataset as DatumaroDataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image as DatumaroImage
from PIL import Image
from rasterio.windows import Window

from AutoDDG.data.datumaro import visualize_random_samples


class AerialSeabirdsProcessor:
    def __init__(self, root_dataset_path="datasets/aerial_seabirds_west_africa"):
        self.root_dataset_path = Path(root_dataset_path)
        self.image_path = self.root_dataset_path / "seabirds_rgb.tif"
        self.labels_path = self.root_dataset_path / "labels_birds_full.csv"
        self.image_chunk_path = self.root_dataset_path / "image_chunks"
        self.label_chunk_path = self.root_dataset_path / "label_chunks"
        self.image_jpg_path = self.root_dataset_path / "image_chunks_jpg"
        self.not_empty_jpg_path = self.root_dataset_path / "label_image_chunks_jpg"
        self.datumaro_output_dir = self.root_dataset_path / "datumaro"
        self.logger = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def convert_spatial_to_pixel(
        x_spatial, y_spatial, im_width_px=27569, im_width_m=292.48, im_height_px=28814
    ):
        """
        Convert spatial coordinates (assumed meters) to pixel coordinates.

        Args:
            x_spatial (float): X coordinate in spatial units (meters).
            y_spatial (float): Y coordinate in spatial units (meters).
            im_width_px (int): Image width in pixels.
            im_width_m (float): Image width in meters.
            im_height_px (int): Image height in pixels.

        Returns:
            tuple: (x_pixel, y_pixel) coordinates.
        """
        conversion_factor = im_width_px / im_width_m
        x_pixel = x_spatial * conversion_factor
        y_pixel = im_height_px - (y_spatial * conversion_factor)
        return x_pixel, y_pixel

    def chunk_image(self, chunk_size=512, overlap=32):
        """
        Chunk the large .tif image into smaller tiles, excluding the last row/column.
        Args:
            chunk_size (int): Size of each chunk (in pixels).
            overlap (int): Overlap between chunks (in pixels).
        """
        self.logger.info("Starting image chunking...")
        self.image_chunk_path.mkdir(parents=True, exist_ok=True)
        try:
            with rasterio.open(self.image_path) as src:
                for j in range(0, src.height - chunk_size + 1, chunk_size - overlap):
                    for i in range(0, src.width - chunk_size + 1, chunk_size - overlap):
                        window = Window(i, j, chunk_size, chunk_size)
                        transform = src.window_transform(window)
                        out_image = src.read(window=window)
                        out_meta = src.meta.copy()

                        # Update metadata for each chunk
                        out_meta.update(
                            {
                                "driver": "GTiff",
                                "height": out_image.shape[1],
                                "width": out_image.shape[2],
                                "transform": transform,
                                "compress": "lzw",
                            }
                        )

                        out_path = self.image_chunk_path / f"{i}_{j}.tif"
                        with rasterio.open(out_path, "w", **out_meta) as dest:
                            dest.write(out_image)
                        self.logger.debug(f"Created image chunk: {out_path}")
            self.logger.info("Image chunking completed successfully.")
        except Exception as e:
            self.logger.error(f"Error during image chunking: {e}", exc_info=True)

    def chunk_labels(self, chunk_size=512, overlap=32):
        """
        Chunk label CSV by converting full-image coords to local chunk coords
        and saving each chunk's labels to its own CSV.
        Args:
            chunk_size (int): Size of each chunk (in pixels).
            overlap (int): Overlap between chunks (in pixels).
        """
        self.logger.info("Starting label chunking...")
        try:
            with rasterio.open(self.image_path) as src:
                image_shape = (src.height, src.width)

            labels = pd.read_csv(self.labels_path)
            labels[["X_pixel", "Y_pixel"]] = labels.apply(
                lambda row: self.convert_spatial_to_pixel(row["X"], row["Y"]),
                axis=1,
                result_type="expand",
            )

            self.label_chunk_path.mkdir(parents=True, exist_ok=True)
            for j in range(0, image_shape[0] - chunk_size + 1, chunk_size - overlap):
                for i in range(
                    0, image_shape[1] - chunk_size + 1, chunk_size - overlap
                ):
                    subset = labels[
                        (labels["X_pixel"] >= i)
                        & (labels["X_pixel"] < i + chunk_size)
                        & (labels["Y_pixel"] >= j)
                        & (labels["Y_pixel"] < j + chunk_size)
                    ].copy()

                    # Adjust to local coords
                    subset["X_pixel"] -= i
                    subset["Y_pixel"] -= j

                    out_path = self.label_chunk_path / f"{i}_{j}.csv"
                    subset.to_csv(out_path, index=False)
                    self.logger.debug(f"Created label chunk: {out_path}")
            self.logger.info("Label chunking completed successfully.")
        except Exception as e:
            self.logger.error(f"Error during label chunking: {e}", exc_info=True)

    def verify_random_chunk(self):
        """
        Pick a random chunk and overlay the label points on the image for manual verification.
        """
        self.logger.info("Starting random chunk verification...")
        try:
            label_files = list(self.label_chunk_path.glob("*.csv"))
            valid_label_files = [f for f in label_files if f.stat().st_size > 100]

            if not valid_label_files:
                self.logger.warning("No valid label files found for verification.")
                return

            selected_label_file = random.choice(valid_label_files)
            selected_image_file = self.image_chunk_path / (
                selected_label_file.stem + ".tif"
            )

            if not selected_image_file.exists():
                self.logger.warning(f"Image file {selected_image_file} does not exist.")
                return

            image = Image.open(selected_image_file)
            labels = pd.read_csv(selected_label_file)

            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            plt.scatter(
                labels["X_pixel"],
                labels["Y_pixel"],
                color="blue",
                s=20,
                label="Birds",
            )
            plt.title(f"Verification: {selected_image_file.name}")
            plt.axis("off")
            plt.legend()
            plt.show()
            self.logger.info(
                f"Displayed verification for chunk: {selected_image_file.name}"
            )
        except Exception as e:
            self.logger.error(f"Error during chunk verification: {e}", exc_info=True)

    def convert_tif_to_jpg(self):
        """
        Convert all .tif files in image_chunk_path to .jpg in image_jpg_path.
        """
        self.logger.info("Starting TIFF to JPG conversion...")
        self.image_jpg_path.mkdir(parents=True, exist_ok=True)
        try:
            for filename in self.image_chunk_path.glob("*.tif"):
                try:
                    img = Image.open(filename)
                    jpg_filename = f"{filename.stem}.jpg"  # Ensure no double .jpg
                    jpg_path = self.image_jpg_path / jpg_filename
                    img.convert("RGB").save(jpg_path, "JPEG")
                    self.logger.debug(f"Converted {filename.name} to {jpg_filename}")
                except Exception as e:
                    self.logger.error(
                        f"Error converting {filename.name} to JPG: {e}", exc_info=True
                    )
            self.logger.info("TIFF to JPG conversion completed successfully.")
        except Exception as e:
            self.logger.error(
                f"Error during TIFF to JPG conversion: {e}", exc_info=True
            )

    def filter_non_empty_csv(self):
        """
        Copy corresponding .jpg images (from image_jpg_path to not_empty_jpg_path)
        if their label CSV has more than 1 row.
        """
        self.logger.info(
            "Starting filtering of non-empty CSVs and copying corresponding JPGs..."
        )
        self.not_empty_jpg_path.mkdir(parents=True, exist_ok=True)
        try:
            for csv_file in self.label_chunk_path.glob("*.csv"):
                df = pd.read_csv(csv_file)
                if len(df) > 1:
                    base = csv_file.stem
                    src_jpg = self.image_jpg_path / f"{base}.jpg"
                    dst_jpg = self.not_empty_jpg_path / f"{base}.jpg"
                    if src_jpg.exists():
                        shutil.copy(src_jpg, dst_jpg)
                        self.logger.debug(f"Copied {src_jpg.name} to {dst_jpg}")
                    else:
                        self.logger.warning(f"Source JPG {src_jpg} does not exist.")
            self.logger.info(
                "Filtering and copying of non-empty JPGs completed successfully."
            )
        except Exception as e:
            self.logger.error(
                f"Error during filtering non-empty CSVs: {e}", exc_info=True
            )

    def export_to_datumaro(
        self, export_format="datumaro", output_dir="exported_dataset"
    ):
        """
        Export the processed dataset to a specified format using Datumaro.

        Args:
            export_format (str): The desired export format (e.g., 'coco', 'yolo', 'voc', etc.).
            output_dir (str): Directory where the exported dataset will be saved.
        """
        self.logger.info(
            f"Exporting dataset to '{export_format}' format using Datumaro..."
        )

        try:
            # 1) Create a Datumaro dataset
            datumaro_dataset = DatumaroDataset(media_type=DatumaroImage)

            # 2) Define label categories
            label_categories = LabelCategories()
            label_categories.add("bird")  # Add as many classes as you need

            # Attach our label categories to the dataset
            datumaro_dataset.categories()[AnnotationType.label] = label_categories

            # 3) Iterate over label CSVs and create Datumaro items
            for csv_file in self.label_chunk_path.glob("*.csv"):
                df = pd.read_csv(csv_file)
                base_filename = csv_file.stem + ".jpg"
                image_path = self.image_jpg_path / base_filename

                if not image_path.exists():
                    self.logger.warning(f"Image {image_path} does not exist. Skipping.")
                    continue

                # Create annotations
                annotations = []
                for _, row in df.iterrows():
                    if pd.notnull(row["X_pixel"]) and pd.notnull(row["Y_pixel"]):
                        x, y = row["X_pixel"], row["Y_pixel"]

                        # Use a Points annotation for (x, y)
                        annotations.append(
                            Points([x, y], label=0)  # index of "bird" class
                        )

                # 4) Create the DatasetItem
                item = DatasetItem(
                    id=base_filename,
                    subset="train",  # or "test"/"val", adjust as needed
                    media=DatumaroImage.from_file(path=str(image_path)),
                    annotations=annotations,
                )
                datumaro_dataset.put(item)
                self.logger.debug(f"Added item to Datumaro dataset: {base_filename}")

            # 5) Export the dataset
            output_dir_path = Path(output_dir)
            output_dir_path.mkdir(parents=True, exist_ok=True)
            datumaro_dataset.export(
                save_dir=str(output_dir_path), format=export_format, save_media=True
            )
            self.logger.info(
                f"Dataset successfully exported to '{output_dir_path}' in '{export_format}' format."
            )
        except Exception as e:
            self.logger.error(f"Error during exporting to Datumaro: {e}", exc_info=True)

    def process_all(
        self,
        verify=False,
        visualize=False,
        chunk_size=512,
        overlap=32,
        export_format="datumaro",
        output_dir="exported_dataset",
        random_seed=42,
    ):
        self.logger.info("Starting processing pipeline...")
        random.seed(random_seed)

        self.logger.info("Step 1: Chunking labels...")
        self.chunk_labels(chunk_size=chunk_size, overlap=overlap)

        self.logger.info("Step 2: Chunking images...")
        self.chunk_image(chunk_size=chunk_size, overlap=overlap)

        if verify:
            self.logger.info("Step 3: Verifying a random chunk...")
            self.verify_random_chunk()

        self.logger.info("Step 4: Converting TIFF chunks to JPG...")
        self.convert_tif_to_jpg()

        self.logger.info(
            "Step 5: Filtering non-empty CSVs and copying corresponding JPGs..."
        )
        self.filter_non_empty_csv()

        self.logger.info("Step 6: Exporting to Datumaro format...")
        self.export_to_datumaro(export_format=export_format, output_dir=output_dir)

        if visualize:
            visualize_random_samples(
                self.datumaro_output_dir, num_samples=9, save_dir=self.root_dataset_path
            )

        self.logger.info("Processing pipeline completed successfully.")
