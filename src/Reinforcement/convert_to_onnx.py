import tensorflow as tf
import tf2onnx
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
load_path = os.path.join(current_dir, 'Models', 'model_v8.keras')
save_path = os.path.join(current_dir, 'Models', 'model_v8.onnx')

model = tf.keras.models.load_model(load_path)
input_signature = [tf.TensorSpec(model.inputs[0].shape, tf.float32, name='input')]

onnx_model, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=input_signature,
    opset=13
)

with open(save_path, 'wb') as f:
    f.write(onnx_model.SerializeToString())

print("Done! model_v8.onnx created.")