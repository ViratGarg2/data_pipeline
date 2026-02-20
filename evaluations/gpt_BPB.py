import torch
import math
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

def calculate_bpb_and_lt_lb(text, model_name="gpt2"):
    """
    Calculate BPB (Bits per Byte) and LT/LB metrics.
    
    Args:
        text: Input text string
        model_name: Hugging Face model name (default: "gpt2")
    
    Returns:
        dict with keys: bpb, lt_lb, total_nll, num_tokens, num_bytes
        
    Where:
        - BPB: Bits per Byte (cross-entropy loss converted to bits per UTF-8 byte)
        - LT/LB: Length in Tokens / Length in Bytes (tokenization efficiency)
    """
    # 1. Load model + tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()

    # 2. Tokenize input
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids
    
    # Count tokens (excluding padding, but including BOS/EOS if present)
    num_tokens = input_ids.shape[1]

    # 3. Get model outputs (logits)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss  # average NLL per token (in nats)
        logits = outputs.logits

    # 4. Compute total NLL manually (teacher forcing)
    # We predict tokens 1 to N given tokens 0 to N-1
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")
    total_nll = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )

    total_nll = total_nll.item()  # in nats
    
    # Number of predicted tokens (N-1 for sequence of length N)
    num_predicted_tokens = num_tokens - 1

    # 5. Count UTF-8 bytes
    num_bytes = len(text.encode("utf-8"))

    # 6. Calculate metrics
    # BPB: Bits per Byte = total_nll_bits / total_bytes
    bpb = total_nll / (num_bytes * math.log(2))
    
    # LT/LB: Length in Tokens / Length in Bytes
    # LT = total number of tokens in the dataset
    # LB = total number of UTF-8 bytes in the dataset
    lt_lb_ratio = num_tokens / num_bytes

    return {
        "bpb": bpb,
        "lt_lb": lt_lb_ratio,
        "loss_per_token": total_nll / num_predicted_tokens,
        "loss_per_byte": total_nll / num_bytes,
        "total_nll": total_nll,
        "num_tokens": num_tokens,
        "num_predicted_tokens": num_predicted_tokens,
        "num_bytes": num_bytes
    }

def calculate_bpb(text, model_name="gpt2"):
    """Backward compatibility: just return BPB value."""
    results = calculate_bpb_and_lt_lb(text, model_name)
    return results["bpb"]

# ---- RUN IT ----
if __name__ == "__main__":
    sample = "Modern language models are evaluated using bits per byte to ensure fair comparison across tokenizers."
    
    # Get all metrics
    results = calculate_bpb_and_lt_lb(sample)
    
    print("=== LANGUAGE MODEL EVALUATION METRICS ===")
    print(f"Text: '{sample}'")
    print(f"UTF-8 bytes: {results['num_bytes']}")
    print(f"Tokens: {results['num_tokens']}")
    print(f"Predicted tokens: {results['num_predicted_tokens']}")
    print()
    print(f"BPB (Bits per Byte): {results['bpb']:.4f} bits/byte")
    print(f"LT/LB (Length Tokens / Length Bytes): {results['lt_lb']:.4f}")
    print()
    print(f"Loss per Token: {results['loss_per_token']:.4f} nats/token")
    print(f"Loss per Byte: {results['loss_per_byte']:.4f} nats/byte")
    print(f"Total NLL: {results['total_nll']:.4f} nats")
    
    # Backward compatibility test
    bpb_only = calculate_bpb(sample)
    print(f"\nBackward compatibility BPB: {bpb_only:.4f} bits/byte")