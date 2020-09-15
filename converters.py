import shutil
from pathlib import Path
import onnx
import tensorflow as tf
from tensorflow.python.keras import backend as K
from onnx2keras import onnx_to_keras


def pytorch2savedmodel(onnx_model_path, saved_model_dir):

    onnx_model = onnx.load(onnx_model_path)

    input_names = ["image_array"]
    k_model = onnx_to_keras(
        onnx_model=onnx_model,
        input_names=input_names,
        change_ordering=True,
        verbose=False,
    )

    weights = k_model.get_weights()

    K.set_learning_phase(0)

    saved_model_dir = Path(saved_model_dir)
    if saved_model_dir.exists():
        shutil.rmtree(str(saved_model_dir))
    saved_model_dir.mkdir()

    tf.saved_model.save(
        k_model,
        str(saved_model_dir.joinpath("1")),
    )


def savedmodel2tflite(
    saved_model_dir, tflite_model_path, quantize=False, representative_dataset=None
):
    saved_model_dir = str(Path(saved_model_dir).joinpath("1"))
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        converter.representative_dataset = representative_dataset
    tflite_model = converter.convert()

    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)

    return tflite_model
