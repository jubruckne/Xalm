# Converts a model consisting of a huggingface config.json, tokenizer.json, and .safetensors weights into a .yalm file,
# which:
# - Normalizes the config to a common format in the header
# - Combines any safetensors shards
# - Reads the token vocabulary into a simpler format
# - Performs quantization to fp8 if specified

import argparse
import os
import json
import sys
from enum import Enum
from urllib.parse import urljoin

import requests
import safetensors
from safetensors.torch import save_file
import torch
from tqdm import tqdm

SUPPORTED_ARCHITECTURES = [
    "MistralForCausalLM"
]

class TensorType(Enum):
    f32 = 1
    f16 = 2
    bf16 = 3
    f8_e4m3 = 4
    f8_e5m2 = 5
    f4_e2m1 = 6

    @staticmethod
    def get_supported_types():
        return [name for name in TensorType.__members__]

    def convert_to(self, t: torch.tensor) -> torch.tensor:
        if self == TensorType.f32:
            converted_tensor = t.to(torch.float32)
        elif self == TensorType.f16:
            converted_tensor = t.to(torch.float16)
        elif self == TensorType.bf16:
            converted_tensor = t.to(torch.bfloat16)
        elif self == TensorType.f8_e4m3:
            torchtype = getattr(torch, "float8_e4m3fn")
            converted_tensor = t.to(torchtype)
        elif self == TensorType.f8_e5m2:
            torchtype = getattr(torch, "float8_e5m2")
            converted_tensor = t.to(torchtype)
        elif self == "f4.e2m1":
            converted_tensor = quantize_to_f4_e2m1(t)
        else:
            raise ValueError(f"Unsupported target dtype: {self}")

        return converted_tensor


class Metadata:
    def __init__(self, config, type_str):
        arch = config["architectures"][0]
        if arch not in SUPPORTED_ARCHITECTURES:
            raise Exception(f"Architecture {arch} is not supported, must be one of {SUPPORTED_ARCHITECTURES}")
        self.arch = arch
        if type_str not in TensorType.get_supported_types():
            raise Exception(f"Data type {type_str} is not supported, must be one of {TensorType.get_supported_types()}")
        self.type = type_str
        if arch == "MistralForCausalLM":
            self.dim = config["hidden_size"]
            self.hidden_dim = config["intermediate_size"]
            self.head_dim = config.get("head_dim", config["hidden_size"] // config["num_attention_heads"])
            self.n_layers = config["num_hidden_layers"]
            self.n_heads = config["num_attention_heads"]
            self.n_kv_heads = config.get("num_key_value_heads", config["num_attention_heads"])
            self.vocab_size = config["vocab_size"]
            self.max_seq_len = config["max_position_embeddings"]
            self.bos_token_id = config["bos_token_id"]
            self.eos_token_id = config["eos_token_id"]
            self.rope_theta = config.get("rope_theta", 10000.0)
            self.rotary_dim = int(self.head_dim * config.get("partial_rotary_factor", 1))
            self.norm_eps = config["rms_norm_eps"]
            self.norm_type = "rmsnorm"

            assert config["hidden_act"] in ["gelu", "silu"]
            self.act_type = config["hidden_act"]

    def to_dict(self):
        result = {"arch": self.arch, "dtype": self.type}
        if self.arch == "MistralForCausalLM":
            result["dim"] = str(self.dim)
            result["hidden_dim"] = str(self.hidden_dim)
            result["head_dim"] = str(self.head_dim)
            result["n_layers"] = str(self.n_layers)
            result["n_heads"] = str(self.n_heads)
            result["n_kv_heads"] = str(self.n_kv_heads)
            result["vocab_size"] = str(self.vocab_size)
            result["max_seq_len"] = str(self.max_seq_len)
            result["bos_token_id"] = str(self.bos_token_id)
            result["eos_token_id"] = str(self.eos_token_id)
            result["rope_theta"] = str(self.rope_theta)
            result["rotary_dim"] = str(self.rotary_dim)
            result["norm_eps"] = str(self.norm_eps)
            result["norm_type"] = str(self.norm_type)
            result["act_type"] = str(self.act_type)
        return result

def load_tokens(tokenizer_path, vocab_size):
    tokens = [""] * vocab_size
    with open(tokenizer_path, "r") as f:
        tokenizer = json.load(f)

    vocab = tokenizer["model"]["vocab"]
    assert len(vocab) <= vocab_size

    for t, i in vocab.items():
        tokens[i] = t

    for added in tokenizer["added_tokens"]:
        tokens[added["id"]] = added["content"]

    # Preprocess tokens into UTF-8 encoding
    for i, t in enumerate(tokens):
        t = t.replace('\u2581', ' ') # sentencepiece uses this character as whitespace
        b = t.encode('utf-8')
        b = b.replace(b"\0", b"\7") # replace null bytes with bell characters
        assert b.count(0) == 0 # no null bytes allowed
        tokens[i] = b

    return tokens

def quantize_to_f4_e2m1(tensor):
    if tensor.dtype != torch.float32:
        raise ValueError("Input tensor must be of dtype float32")

    # Extract sign
    sign_bit = (tensor < 0).to(torch.int32)
    abs_tensor = torch.abs(tensor)

    # Compute exponent (bias = 1) and clamp range
    exponent = torch.floor(torch.log2(abs_tensor)).clamp(-1, 2)
    exponent_bits = (exponent + 1).to(torch.int32)

    # Normalize and compute mantissa
    normalized = abs_tensor / (2 ** exponent)
    mantissa = ((normalized - 1.0) * 2).clamp(0, 1).to(torch.int32)

    # Combine to form FP4
    fp4 = (sign_bit << 3) | (exponent_bits << 1) | mantissa

    # Pack FP4 into int8 (two FP4 values per int8)
    packed = (fp4[::2] << 4) | fp4[1::2]
    return packed.to(torch.int8)

def dequantize_from_f4_e2m1(packed_tensor):
    # Unpack int8 into two FP4 values each
    fp4_high = (packed_tensor >> 4) & 0x0F
    fp4_low = packed_tensor & 0x0F

    # Concatenate unpacked FP4 values
    fp4 = torch.cat([fp4_high, fp4_low], dim=0)

    # Extract components
    sign_bit = (fp4 >> 3) & 0b1
    exponent_bits = (fp4 >> 1) & 0b11
    mantissa_bits = fp4 & 0b1

    # Decode values
    sign = (-1) ** sign_bit
    exponent = exponent_bits - 1  # Remove bias
    mantissa = mantissa_bits / 2
    value = sign * (1 + mantissa) * (2 ** exponent)
    return value.to(torch.float32)


def load_weights(model_files, target_type: TensorType, metadata, tie_word_embeddings):
    """
    Load all weights from the model files in huggingface format into a dictionary of tensors,
    normalizing the attention weights, and casting all tensors (except for all layer norm weights,
    which are converted to float32) to the specified dtype.
    """
    weights = {}
    for model_path in model_files:
        ext = os.path.splitext(model_path)[1]
        if ext == ".safetensors":
            with safetensors.safe_open(model_path, framework="pt") as f:
                for k in f.keys():
                    assert(k not in weights)
                    weights[k] = f.get_tensor(k)

    # Stolen from https://github.com/zeux/calm/blob/86dfb56daba5052c260a2dd86d296309cfbd4324/tools/convert.py#L223
    # huggingface permutes WQ and WK, this function reverses it
    # see https://github.com/huggingface/transformers/blob/b132c1703eb1c8bd9dfa4ad6a9be2bfd6ef819e9/src/transformers/models/llama/convert_llama_weights_to_hf.py#L122
    def permute_reverse(w, heads, rotary_dim):
        head_dim = w.shape[0] // heads
        assert rotary_dim <= head_dim
        w = torch.unflatten(w, 0, (-1, head_dim))
        # wr is the rotary part, wk is the part kept un-rotated
        wr = w[:, :rotary_dim]
        wk = w[:, rotary_dim:]
        # switch wr from outputting two rotary_dim/2 chunks to outputting values interleaved
        wr = torch.unflatten(wr, 1, (2, -1))
        wr = wr.transpose(1, 2)
        wr = wr.flatten(1, 2)
        # assemble the heads back
        w = torch.cat([wr, wk], dim=1)
        return torch.flatten(w, 0, 1)

    # convert weights
    progress = 0
    def conv(name: str, t: torch.Tensor) -> torch.Tensor:
        nonlocal progress
        progress += 1
        actual_type = target_type
        if name == "model.embed.weight" or name == "model.output.weight":
            actual_type = TensorType.f16

        if args.analyze:
            print(f"{name}[{t.dtype}, range={torch.max(t) - torch.min(t):.4f}]", end="")
        else:
            print(f"{name}: {t.dtype} => {actual_type}")

        convt = actual_type.convert_to(t)

        if args.analyze:
            o = t.to(torch.float32)
            q = convt.to(torch.float32)
            mse = torch.mean((o * 1000 - q * 1000) ** 2)
            normalized_error = torch.sum(torch.abs(o - q)) / torch.sum(torch.abs(o))
            cos_sim = torch.cosine_similarity(o.view(-1).unsqueeze(0), q.view(-1).unsqueeze(0)).item()
            snr = 10 * torch.log10(torch.sum(o ** 2) / torch.sum((o - q) ** 2))
            accuracy = torch.mean((torch.abs(o - q) <= 0.00025).float()).item()

            print(f" => {actual_type}: mse={mse:.2f}, norm_err_k={normalized_error:.2f}, cosine={cos_sim:.3f}, snr={snr:.2f}, accuracy={accuracy:.2f})")

        return convt

    tensors = {}

    tensors["model.embed.weight"] = conv("model.embed.weight", weights["model.embed_tokens.weight"])

    for l in range(config["num_hidden_layers"]):
        tensors[f"model.layers.{l}.attn.norm.weight"] = weights[f"model.layers.{l}.input_layernorm.weight"].float()

        rotary_dim = metadata.rotary_dim
        n_heads = metadata.n_heads
        n_kv_heads = metadata.n_kv_heads

        tensors[f"model.layers.{l}.attn.wq.weight"] = conv(f"model.layers.{l}.attn.wq.weight", permute_reverse(weights[f"model.layers.{l}.self_attn.q_proj.weight"], n_heads, rotary_dim))
        tensors[f"model.layers.{l}.attn.wk.weight"] = conv(f"model.layers.{l}.attn.wk.weight", permute_reverse(weights[f"model.layers.{l}.self_attn.k_proj.weight"], n_kv_heads, rotary_dim))

        tensors[f"model.layers.{l}.attn.wv.weight"] = conv(f"model.layers.{l}.attn.wv.weight", weights[f"model.layers.{l}.self_attn.v_proj.weight"])
        tensors[f"model.layers.{l}.attn.wo.weight"] = conv(f"model.layers.{l}.attn.wo.weight", weights[f"model.layers.{l}.self_attn.o_proj.weight"])

        tensors[f"model.layers.{l}.mlp.norm.weight"] = weights[f"model.layers.{l}.post_attention_layernorm.weight"].float()

        tensors[f"model.layers.{l}.mlp.w1.weight"] = conv(f"model.layers.{l}.mlp.w1.weight", weights[f"model.layers.{l}.mlp.gate_proj.weight"])
        tensors[f"model.layers.{l}.mlp.w2.weight"] = conv(f"model.layers.{l}.mlp.w2.weight", weights[f"model.layers.{l}.mlp.down_proj.weight"])
        tensors[f"model.layers.{l}.mlp.w3.weight"] = conv(f"model.layers.{l}.mlp.w3.weight", weights[f"model.layers.{l}.mlp.up_proj.weight"])

    tensors["model.norm.weight"] = weights["model.norm.weight"].float()
    if not tie_word_embeddings:
        tensors["model.output.weight"] = conv("model.output.weight", weights["lm_head.weight"])

    return tensors

def download_file(url: str, output_path: str, token: str = None):
    """
    Download a file from the given URL and save it to the specified path with a progress bar.
    """
    print(f"Starting download: {url}")

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    response = requests.get(url, stream=True, headers=headers)

    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        with open(output_path, 'wb') as f, tqdm(
                desc=f"{output_path}",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        print(f"Saved to {output_path}")
    else:
        raise Exception(f"Failed to download {url}. HTTP Status: {response.status_code}")


def download_model_files(base_url: str, output_dir: str):
    """
    Download model weights, tokenizer, and configuration files from a base URL.

    Args:
        base_url (str): The base URL hosting the model files.
        output_dir (str): Local directory to save the downloaded files.

    Returns:
        dict: Paths to the downloaded model files.
    """
    files_to_download = [
        "tokenizer.json",        # Tokenizer file
        "config.json",           # Model configuration
        "model-00001-of-00003.safetensors",  # Example weights part 1
        "model-00002-of-00003.safetensors",  # Example weights part 2
        "model-00003-of-00003.safetensors",  # Example weights part 3
    ]

    downloaded_files = {}
    os.makedirs(output_dir, exist_ok=True)

    for file_name in files_to_download:
        file_url = urljoin(base_url, file_name)
        file_path = os.path.join(output_dir, file_name)
        download_file(file_url, file_path)
        downloaded_files[file_name] = file_path

    return downloaded_files

def process_input(input_path: str, temp_dir: str, token: str = None) -> dict:
    """
    Process the input parameter: if it's a URL, download files; otherwise, directly use local files.

    Args:
        input_path (str): URL or local path.
        temp_dir (str): temporary directory to save the downloaded files.

    Returns:
        dict: Paths to the processed model files.
    """
    files_to_process = [
        "tokenizer.json",        # Tokenizer file
        "config.json",           # Model configuration
        "model-00001-of-00003.safetensors",  # Example weights part 1
        "model-00002-of-00003.safetensors",  # Example weights part 2
        "model-00003-of-00003.safetensors",  # Example weights part 3
    ]

    processed_files = {}
    if input_path.startswith("http://") or input_path.startswith("https://"):
        # Download files from URL
        # https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/raw/main/tokenizer.json
        # https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/resolve/main/model-00001-of-00003.safetensors

        os.makedirs(temp_dir, exist_ok=True)
        for file_name in files_to_process:
            if file_name == "tokenizer.json" or file_name == "config.json":
                file_url = urljoin(input_path + "raw/main/", file_name)
            else:
                file_url = urljoin(input_path + "resolve/main/", file_name)

            file_path = os.path.join(temp_dir, file_name)
            download_file(file_url, file_path, token)
            processed_files[file_name] = file_path
    else:
        # Use local files without copying
        for file_name in files_to_process:
            source_path = os.path.join(input_path, file_name)
            if os.path.exists(source_path):
                print(f"Using local file: {source_path}")
                processed_files[file_name] = source_path
            else:
                raise Exception(f"Missing required file: {source_path}")

    return processed_files

def validate_files(file_paths: dict):
    """
    Ensure all required files were downloaded successfully.

    Args:
        file_paths (dict): Dictionary of expected files and their paths.

    Raises:
        Exception: If any file is missing.
    """
    for name, path in file_paths.items():
        if not os.path.exists(path):
            raise Exception(f"Missing required file: {name} at {path}")
    print("All files downloaded and validated successfully!")


def convert_tensor(original_tensor: torch.Tensor, dtype_str: str) -> torch.Tensor:
    """
    Convert tensor data to the target_dtype.
    """

    if original_tensor.device.type != "cpu":
        raise ValueError(f"Tensor must be on cpu!")

    converted_tensor: torch.Tensor

    if dtype_str == "f32":
        converted_tensor = original_tensor.to(torch.float32)
    elif dtype_str == "f16":
        converted_tensor = original_tensor.to(torch.float16)
    elif dtype_str == 'f8.e4m3':
        torchtype = getattr(torch, "float8_e4m3fn")
        converted_tensor = original_tensor.to(torchtype)
    elif dtype_str == 'f8.e5m2':
        torchtype = getattr(torch, "float8_e5m2")
        converted_tensor = original_tensor.to(torchtype)
    elif dtype_str == "f4.e2m1":
        converted_tensor = quantize_to_f4_e2m1(original_tensor)
    else:
        raise ValueError(f"Unsupported target dtype: {dtype_str}")

    # print(f"Converted tensor: {converted_tensor}")
    return converted_tensor

        #
        # converted = torch.empty(tensor.numel(), dtype=torch.uint8, device=tensor.device)
        #
        # chunk_size = max(1024, tensor.numel() // 128)  # At least 1MB or 1% of tensor size
        #
        # with tqdm(total=tensor.numel(), desc=f"Converting {name} {tuple(tensor.shape)}", unit="values") as progress:
        #     for start in range(0, tensor.numel(), chunk_size):
        #         end = min(start + chunk_size, tensor.numel())
        #         chunk = tensor.flatten()[start:end]  # Flatten and slice the tensor
        #         converted[start:end] = float32_to_f8_e4m3(chunk)  # Perform the conversion
        #         progress.update(end - start)
        #
        # if tensor.numel() != converted.numel():
        #     raise ValueError(f"Unexpected element count!")
        #
        # print(tensor.storage_type())
        # print(converted.storage_type())find
        #
        # # Reshape to the original tensor shape
        # return converted.reshape(tensor.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert model files with tokenizer and config, supporting URL input.")

    parser.add_argument("--input", type=str, help="Base URL/path with files.", required=True)
    parser.add_argument("--output", type=str, help="Output file path for the converted model.", default = "", required=False)
    parser.add_argument("--temp-dir", type=str, help="Temporary directory to store downloaded files.", default="./temp")
    parser.add_argument("--type", type=str, default="f16", choices=TensorType.get_supported_types())
    parser.add_argument("--token", type=str, help="Hugging Face access token for private models.", required=False)
    parser.add_argument("--analyze", type=bool, nargs="?", const=True, help="Only analyze the input file.", default=False)

    args = parser.parse_args()

    if args.output == "":
        if not args.analyze:
            print(f"--output must be specified!")
            sys.exit(1)

    try:
        print("Processing input...")
        processed_files = process_input(args.input, args.temp_dir, args.token)
        validate_files(processed_files)
    except Exception as e:
        print(f"Error processing input: {e}")
        sys.exit(1)

    print(torch.__version__)

    model_weights = [processed_files[f"model-0000{i}-of-00003.safetensors"] for i in range(1, 4)]
    tokenizer_path = processed_files["tokenizer.json"]
    config_path = processed_files["config.json"]

    # Placeholder for conversion logic
    print(f"Using config: {config_path}")
    print(f"Using tokenizer: {tokenizer_path}")
    print(f"Using weights: {model_weights}")

    with open(config_path, "r") as f:
        config = json.load(f)
        metadata = Metadata(config, args.type)

    tokens = load_tokens(tokenizer_path, metadata.vocab_size)
    tensors = load_weights(model_weights, TensorType[args.type], metadata, config.get("tie_word_embeddings", None))

    # add tokenizer tensors at the end (to maximize the chance of model tensor alignment)
    # note: we concatenate all bytes of all tokens into a single tensor
    tensors["tokenizer.tokens"] = torch.cat([torch.tensor([x for x in b] + [0], dtype=torch.uint8) for b in tokens])

    if not args.analyze:
        print(f"Saving {len(tensors)} tensors...")
        for k, v in tensors.items():
            assert v.layout == torch.strided and v.is_contiguous()

        save_file(tensors, args.output, metadata.to_dict())

        print(f"saved to {args.output}")

    print("Done!")