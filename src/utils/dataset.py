import os
from PIL import Image
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data_dir, tokenizer, size=512):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.size = size
        self.image_paths = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".png")
        ]
        self.image_paths.sort()  # Ensure consistent ordering

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        text_path = image_path.replace(".png", ".txt")

        # Load and process the image
        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.size, self.size), Image.LANCZOS)
        pixel_values = (
            torch.tensor(list(image.getdata()), dtype=torch.float32).reshape(
                3, self.size, self.size
            )
            / 127.5
            - 1.0
        )

        # Load and process the text caption
        with open(text_path, "r", encoding="utf-8") as f:
            caption = f.read().strip()

        # Tokenize the caption
        inputs = self.tokenizer(
            caption,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": inputs.input_ids[0],
        }
