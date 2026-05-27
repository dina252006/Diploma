from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model_architecture import (
    create_simple_cnn_model,
    create_improved_cnn_model,
    create_resnet18_model,
    create_lwvit_model,
    create_pfvit_model,
)


MODEL_FACTORIES = {
    "Simple CNN": create_simple_cnn_model,
    "Improved CNN": create_improved_cnn_model,
    "ResNet18": create_resnet18_model,
    "LW-ViT": create_lwvit_model,
    "PF-ViT": create_pfvit_model,
}


def fix_chinese_label(label):
    """
    Fix class labels that were stored as UTF-8 text but decoded as cp437.
    Correct labels are returned unchanged.
    """
    if not isinstance(label, str):
        return str(label)

    try:
        fixed_label = label.encode("cp437").decode("utf-8")
        has_chinese = any("\u4e00" <= char <= "\u9fff" for char in fixed_label)

        if has_chinese:
            return fixed_label

    except (UnicodeEncodeError, UnicodeDecodeError):
        pass

    return label


class ChineseCharacterRecognizer:
    def __init__(self, model_name, model_path, device=None):
        self.model_name = model_name
        self.model_path = Path(model_path)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if self.model_name not in MODEL_FACTORIES:
            raise ValueError(f"Unsupported model: {self.model_name}")

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint was not found: {self.model_path}")

        self.checkpoint = self._load_checkpoint()
        self.num_classes = int(self.checkpoint.get("num_classes", 246))
        self.image_size = int(self.checkpoint.get("image_size", 64))

        self.class_to_idx = self._load_class_mapping()
        self.idx_to_class = {
            class_index: class_name
            for class_name, class_index in self.class_to_idx.items()
        }

        self.model = self._load_model()
        self.transform = self._build_transform()

    def _load_checkpoint(self):
        try:
            return torch.load(
                self.model_path,
                map_location=self.device,
                weights_only=False
            )
        except TypeError:
            return torch.load(
                self.model_path,
                map_location=self.device
            )

    def _load_class_mapping(self):
        if "class_to_idx" not in self.checkpoint:
            raise KeyError("The checkpoint does not contain class_to_idx.")

        raw_class_to_idx = self.checkpoint["class_to_idx"]

        fixed_class_to_idx = {
            fix_chinese_label(class_name): int(class_index)
            for class_name, class_index in raw_class_to_idx.items()
        }

        return fixed_class_to_idx

    def _load_model(self):
        model_factory = MODEL_FACTORIES[self.model_name]
        model = model_factory(num_classes=self.num_classes)

        if "model_state_dict" not in self.checkpoint:
            raise KeyError("The checkpoint does not contain model_state_dict.")

        model.load_state_dict(self.checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        return model

    def _build_transform(self):
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])

    def preprocess_image(self, image):
        if not isinstance(image, Image.Image):
            image = Image.open(image)

        image = image.convert("L")
        image_tensor = self.transform(image)
        image_tensor = image_tensor.unsqueeze(0)

        return image_tensor.to(self.device)

    def predict(self, image, top_k=5):
        image_tensor = self.preprocess_image(image)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)

            top_probabilities, top_indices = torch.topk(
                probabilities,
                k=top_k,
                dim=1
            )

        top_probabilities = top_probabilities.squeeze(0).cpu().numpy()
        top_indices = top_indices.squeeze(0).cpu().numpy()

        predicted_index = int(top_indices[0])
        predicted_character = self.idx_to_class[predicted_index]
        confidence = float(top_probabilities[0])

        top_predictions = []

        for rank, (class_index, probability) in enumerate(
            zip(top_indices, top_probabilities),
            start=1
        ):
            class_index = int(class_index)

            top_predictions.append({
                "rank": rank,
                "character": self.idx_to_class[class_index],
                "class_index": class_index,
                "confidence": float(probability)
            })

        return {
            "predicted_character": predicted_character,
            "confidence": confidence,
            "top_predictions": top_predictions
        }

    def get_model_info(self):
        return {
            "model_name": self.checkpoint.get("model_name", self.model_name),
            "checkpoint_path": str(self.model_path),
            "image_size": self.image_size,
            "num_classes": self.num_classes,
            "best_validation_accuracy": self.checkpoint.get("best_validation_accuracy"),
            "best_validation_f1": self.checkpoint.get("best_validation_f1")
        }

    def get_checkpoint_metrics(self):
        return self.get_model_info()
