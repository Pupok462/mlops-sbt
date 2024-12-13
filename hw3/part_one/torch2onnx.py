import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import onnxruntime as ort
import numpy as np
from typing import Any
from thop import profile


class GPT5(nn.Module):

    def __init__(self, N, model_name):
        super(GPT5, self).__init__()
        self.transformer = AutoModel.from_pretrained(
            model_name,
            return_dict=True,
            output_hidden_states=True
        )
        hidden_size = self.transformer.config.hidden_size
        self.fc = nn.Linear(hidden_size, N)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        proj = self.fc(outputs.last_hidden_state)
        return proj


def encode(tokenizer: Any, text: str):
    encoding = tokenizer(text, return_tensors='pt')

    input_ids = torch.tensor(
        encoding["input_ids"],
        dtype=torch.int32,
        device="cpu"
    )
    attention_mask = torch.tensor(
        encoding["attention_mask"], dtype=torch.int32, device="cpu"
    )
    return input_ids, attention_mask


def compute_flops(model, input_ids, attention_mask):
    macs, params, layer_info = profile(model, inputs=(input_ids, attention_mask), ret_layer_info=True)
    total_flops = macs * 2
    print(f"Общее количество FLOPs (умножение-сложение): {total_flops}")
    print(f"Общее количество параметров: {params}")

    for layer, (layer_macs, _, _) in layer_info.items():
        layer_flops = 2 * layer_macs
        print(f"Слой: {layer}, FLOPs: {layer_flops}")


def main():
    model_name = 'dslim/bert-base-NER'
    onnx_filename = 'models/model.onnx'

    # Tokenize text
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_ids, attention_mask = encode(tokenizer, "My name is Clara and I live in Berkeley, California.")

    # Prepare Model
    N = 16
    model = GPT5(N, model_name)
    model.eval()

    # Convert to ONNX
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        onnx_filename,
        opset_version=19,
        input_names=['INPUT_IDS', 'ATTENTION_MASK'],
        output_names=['PROJECTION'],
        dynamic_axes={
            'INPUT_IDS': {0: 'BATCH_SIZE', 1: 'SEQUENCE_LENGTH'},
            'ATTENTION_MASK': {0: 'BATCH_SIZE', 1: 'SEQUENCE_LENGTH'},
            'PROJECTION': {0: 'BATCH_SIZE', 1: 'SEQUENCE_LENGTH'}
        },
    )

    torch_output = model(input_ids, attention_mask).detach().numpy()

    ort_inputs = {
        "INPUT_IDS": input_ids.numpy(),
        "ATTENTION_MASK": attention_mask.numpy()
    }
    ort_session = ort.InferenceSession(onnx_filename)
    ort_outputs = ort_session.run(None, ort_inputs)[0]

    # !!! Sanity Check!!!
    assert np.allclose(torch_output, ort_outputs, atol=1e-5)
    compute_flops(model, input_ids, attention_mask)

    tokenizer.save_pretrained(model_name)


if __name__ == "__main__":
    main()
