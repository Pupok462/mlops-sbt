import numpy as np
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
from tritonclient.utils import np_to_triton_dtype

def call_triton(input_text: str):

    client = InferenceServerClient(url="localhost:8000")
    text = np.array([input_text], dtype=object)

    input_text_tensor = InferInput(
        name="TEXT", shape=text.shape, datatype=np_to_triton_dtype(text.dtype)
    )
    input_text_tensor.set_data_from_numpy(text, binary_data=True)


    outputs = [
        InferRequestedOutput("PROJECTION_ONNX"),
        InferRequestedOutput("PROJECTION_BEST"),
        InferRequestedOutput("PROJECTION_FP16"),
        InferRequestedOutput("PROJECTION_FP32"),
        InferRequestedOutput("PROJECTION_INT8")
    ]

    response = client.infer(
        "ensemble",
        [input_text_tensor],
        outputs=outputs,
    )

    emb_onnx = response.as_numpy("PROJECTION_ONNX")
    emb_best = response.as_numpy("PROJECTION_BEST")
    emb_fp16 = response.as_numpy("PROJECTION_FP16")
    emb_fp32 = response.as_numpy("PROJECTION_FP32")
    emb_int8 = response.as_numpy("PROJECTION_INT8")

    return emb_onnx, emb_best, emb_fp16, emb_fp32, emb_int8

def check_quality(input_text: str):
    emb_onnx, emb_best, emb_fp16, emb_fp32, emb_int8 = call_triton(input_text)
    def mae(a, b):
        return np.mean(np.abs(a - b))

    deviation_best = mae(emb_best, emb_onnx)
    deviation_fp16 = mae(emb_fp16, emb_onnx)
    deviation_fp32 = mae(emb_fp32, emb_onnx)
    deviation_int8 = mae(emb_int8, emb_onnx)

    return deviation_best, deviation_fp16, deviation_fp32, deviation_int8

def main():
    texts = [
        "Hello world",
        "GAY GAY GAY GAY",
        "LOL OLO OLO LOL OLO LOL OLO LOL OLLO",
        "CHILLGUYCHILLGUYCHILLGUYCHILLGUYCHILLGUY",
        "Niga stefi you giki blyak eki yo"
    ]

    deviations_best = []
    deviations_fp16 = []
    deviations_fp32 = []
    deviations_int8 = []

    for t in texts:
        best, fp16, fp32, int8 = check_quality(t)
        deviations_best.append(best)
        deviations_fp16.append(fp16)
        deviations_fp32.append(fp32)
        deviations_int8.append(int8)

    mean_best = np.mean(deviations_best)
    mean_fp16 = np.mean(deviations_fp16)
    mean_fp32 = np.mean(deviations_fp32)
    mean_int8 = np.mean(deviations_int8)

    print(f"Среднее отклонение BEST от ONNX: {mean_best}")
    print(f"Среднее отклонение FP16 от ONNX: {mean_fp16}")
    print(f"Среднее отклонение FP32 от ONNX: {mean_fp32}")
    print(f"Среднее отклонение INT8 от ONNX: {mean_int8}")

if __name__ == "__main__":
    main()
