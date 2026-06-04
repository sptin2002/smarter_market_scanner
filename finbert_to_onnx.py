import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def convert_finbert_to_onnx(local_dir="finbert-local", output_name="finbert.onnx"):
    print(f"[*] Loading PyTorch model from {local_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(local_dir, local_files_only=True)
    model.eval()

    # FIX: Explicitly pad the dummy text to a fixed 128 tokens
    dummy_text = "Apple reports record breaking quarterly revenues and earnings per share."
    inputs = tokenizer(dummy_text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")

    # FIX: Track only batch_size as dynamic. Locking the sequence_length 
    # to 128 stops the internal attention layers from misaligning.
    dynamic_axes = {
        "input_ids": {0: "batch_size"},
        "attention_mask": {0: "batch_size"},
        "token_type_ids": {0: "batch_size"},
        "logits": {0: "batch_size"}
    }

    output_path = os.path.join(local_dir, output_name)
    print(f"[*] Exporting model to ONNX format at: {output_path}...")
    
    with torch.no_grad():
        torch.onnx.export(
            model,
            args=(inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"]),
            f=output_path,
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=14,
            dynamo=False
        )
    print("[+] Export successful! Model successfully locked at sequence length 128.")

if __name__ == "__main__":
    convert_finbert_to_onnx()