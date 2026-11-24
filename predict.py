"""Predict the class of one brain MRI image with the saved final model.
Educational research prototype - NOT a clinically validated diagnostic system.
Usage: python predict.py path/to/mri.jpg [--no-plot]
"""
import argparse, json
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image

MODEL_DIR = Path(__file__).parent / "final"

try:
    _reg = keras.saving.register_keras_serializable(package="btc")
except AttributeError:
    _reg = keras.utils.register_keras_serializable(package="btc")

@_reg
class InputPreprocess(keras.layers.Layer):  # identical to the training definition
    def __init__(self, mode="builtin", mean=0.0, std=1.0, **kwargs):
        kwargs.setdefault("dtype", "float32"); super().__init__(**kwargs)
        self.mode, self.mean, self.std = mode, float(mean), float(std)
    def call(self, x):
        x = tf.cast(x, tf.float32)
        if self.mode == "caffe":   x = tf.reverse(x, axis=[-1]) - tf.constant([103.939, 116.779, 123.68], tf.float32)
        elif self.mode == "torch": x = (x / 255.0 - tf.constant([0.485, 0.456, 0.406], tf.float32)) / tf.constant([0.229, 0.224, 0.225], tf.float32)
        elif self.mode == "standardize": x = (x / 255.0 - self.mean) / self.std
        return x
    def compute_output_shape(self, input_shape): return input_shape
    def get_config(self):
        c = super().get_config(); c.update({"mode": self.mode, "mean": self.mean, "std": self.std}); return c

def preprocess(path, size):
    """Same steps as training: grayscale, centred black pad to square, bicubic resize, 3 identical channels."""
    with Image.open(path) as im:
        im.load(); g = im.convert("L")
    w, h = g.size; s = max(w, h)
    if w != h:
        canvas = Image.new("L", (s, s), 0); canvas.paste(g, ((s - w) // 2, (s - h) // 2)); g = canvas
    a = np.asarray(g.resize((size, size), Image.BICUBIC), dtype=np.uint8)
    return np.repeat(a[..., None], 3, axis=2)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("image"); ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    labels = json.loads((MODEL_DIR / "class_labels.json").read_text())
    names = [labels["index_to_display"][str(i)] for i in range(len(labels["index_to_display"]))]
    size = int(json.loads((MODEL_DIR / "preprocessing_config.json").read_text())["img_size"])
    model = keras.models.load_model(MODEL_DIR / "final_model.keras", custom_objects={"InputPreprocess": InputPreprocess}, compile=False)
    p = model(preprocess(args.image, size)[None].astype(np.float32), training=False).numpy()[0].astype(np.float64)
    p /= p.sum(); k = int(p.argmax())
    print(f"Predicted: {names[k]} (confidence {100 * p[k]:.1f}%)")
    for n, v in zip(names, p): print(f"  {n:<11}{100 * v:6.1f}%")
    print("Educational research prototype - not a clinically validated diagnostic system.")
    if not args.no_plot:
        import matplotlib.pyplot as plt
        with Image.open(args.image) as im: shown = np.asarray(im.convert("L"))
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
        ax[0].imshow(shown, cmap="gray"); ax[0].axis("off"); ax[0].set_title(f"{names[k]} ({100 * p[k]:.1f}%)")
        ax[1].barh(names[::-1], p[::-1]); ax[1].set_xlim(0, 1); ax[1].set_xlabel("Probability")
        plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
