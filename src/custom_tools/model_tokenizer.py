from transformers import AutoTokenizer

def get_tokenizer(model_name):
    match model_name:
        case "Qwen":
            model_path = "/mnt/compression/pytorch/qwen3-8b"
        case "Kimina-Autoformalizer":
            model_path = "/mnt/compression/pytorch/Kimina-Autoformalizer-7B"
        case "Kimina-Prover":
            model_path = "/mnt/compression/pytorch/Kimina-Prover-Preview-Distill-7B"
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
