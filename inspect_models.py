import tensorflow as tf

models = [
    "models/face_model.tflite",
    "models/FaceAntiSpoofing.tflite",
]

for path in models:
    print("\n" + "=" * 70)
    print(path)
    print("=" * 70)

    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()

    print("\nINPUTS:")
    for x in interpreter.get_input_details():
        print("  name:", x["name"])
        print("  shape:", x["shape"])
        print("  dtype:", x["dtype"])
        print("  quantization:", x["quantization"])

    print("\nOUTPUTS:")
    for x in interpreter.get_output_details():
        print("  name:", x["name"])
        print("  shape:", x["shape"])
        print("  dtype:", x["dtype"])
        print("  quantization:", x["quantization"])
