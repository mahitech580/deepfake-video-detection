"""Convert the repository's trained Keras model into a compact LiteRT artifact at build time."""
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "deepfake_model.h5"
TARGET = ROOT / "deepfake_model.tflite"

if not SOURCE.is_file() or SOURCE.stat().st_size < 1_000_000:
    raise SystemExit("Trained deepfake_model.h5 is missing; ensure Git LFS is fetched during the Render build.")

try:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except RuntimeError:
    pass

print(f"Loading trained model ({SOURCE.stat().st_size:,} bytes)", flush=True)
model = tf.keras.models.load_model(str(SOURCE), compile=False)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
print("Converting trained weights to LiteRT/TFLite ...", flush=True)
artifact = converter.convert()
TARGET.write_bytes(artifact)

# Validate the exact artifact that will be used by the runtime.
interpreter = tf.lite.Interpreter(model_path=str(TARGET), num_threads=1)
interpreter.allocate_tensors()
input_info = interpreter.get_input_details()[0]
output_info = interpreter.get_output_details()[0]
shape = tuple(int(v) for v in input_info["shape"])
if shape != (1, 224, 224, 3):
    raise SystemExit(f"Unexpected model input shape: {shape}")
interpreter.set_tensor(input_info["index"], np.zeros(shape, dtype=input_info["dtype"]))
interpreter.invoke()
prediction = np.asarray(interpreter.get_tensor(output_info["index"]))
if prediction.size != 1 or not np.isfinite(prediction).all():
    raise SystemExit("TFLite model smoke test failed.")
print(f"Validated {TARGET.name}: {TARGET.stat().st_size:,} bytes; sample output={float(prediction.reshape(-1)[0]):.6f}", flush=True)
