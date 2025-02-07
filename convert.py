# Converts a model consisting of a huggingface config.json, tokenizer.json, and .safetensors weights into a .xalm file,
# which:
# - Normalizes the config to a common format in the header
# - Combines any safetensors shards
# - Reads the token vocabulary into a simpler format
# - Performs quantization to fp8 if specified

import argparse
import ctypes
import gc
import os
import json
import re
import struct
import sys
from collections import OrderedDict
from enum import Enum
from urllib.parse import urljoin

import xxhash
from torch.fx.experimental.unification.multipledispatch.utils import typename

from quants import quantize as gguf_quantize, dequantize as gguf_dequantize, GGMLQuantizationType
import requests
import safetensors
from safetensors.torch import save_file
import torch
from tqdm import tqdm

SUPPORTED_ARCHITECTURES = [
    "MistralForCausalLM",
    "LlamaForCausalLM"
]

def align_offset(offset, alignment=32):
    """Round up offset to the nearest multiple of alignment."""
    return (offset + (alignment - 1)) // alignment * alignment

def save_binary(filename: str, tensors: OrderedDict, metadata: dict):
    metadata.tensors = sort_tensors(metadata.tensors)

    current_offset = 0

    for tensor_name, meta in metadata.tensors.items():
        current_offset = align_offset(current_offset)
        t: torch.Tensor = tensors[tensor_name]
        storage = t.untyped_storage()
        data_ptr = storage.data_ptr()
        nbytes = storage.nbytes()
        raw_data = (ctypes.c_ubyte * nbytes).from_address(data_ptr)
        hash_value = xxhash.xxh3_64(raw_data)
        meta["hash"] = hash_value.intdigest()
        meta["offset"] = current_offset
        meta["size"] = nbytes
        current_offset += nbytes

    with open(filename, "wb") as f:
        # Convert metadata to JSON
        metadata_json = json.dumps(metadata).encode("utf-8")
        metadata_size = len(metadata_json)

        f.write(struct.pack("Q", metadata_size))
        f.write(metadata_json)

        # Track the current file offset
        metadata_size = f.tell() + 128

        metadata_size = align_offset(metadata_size)
        padding = metadata_size - metadata_size
        if padding > 0:
            f.write(b"\x00" * padding)

        metadata_size = f.tell() - 1
        f.seek(0)
        f.write(struct.pack("Q", metadata_size))
        f.seek(metadata_size)

        # Write tensor data sequentially with alignment
        for tensor_name, tensor in tensors.items():
            # Ensure alignment
            aligned_offset = align_offset(current_offset)
            padding = aligned_offset - current_offset
            if padding > 0:
                f.write(b"\x00" * padding)

            # Update the offset after padding
            current_offset = aligned_offset

            storage = tensor.untyped_storage()
            data_ptr = storage.data_ptr()
            nbytes = storage.nbytes()
            raw_data = (ctypes.c_ubyte * nbytes).from_address(data_ptr)

            hash_value = xxhash.xxh3_64(raw_data)
            metadata[tensor_name]["hash"] = hash_value.intdigest()


class XTensor:
    def __init__(self, t: torch.Tensor, name: str):
        self.name = name
        self.t = t
        self.type = XType.from_torch(t.dtype)

class XType(Enum):
    f32 = 1
    f16 = 2
    bf16 = 3
    f8_e4m3 = 4
    f8_e5m2 = 5
    f4_e2m1 = 6
    qi8 = 7
    qi4 = 8
    qi3 = 9

    # gguf types
    q4_0 = 1007
    q4_1 = 1008
    q5_0 = 1009
    q5_1 = 1010
    q8_0 = 1011
    tq1_0 = 1012

    @staticmethod
    def get_supported_types():
        return [name for name in XType.__members__]

    def name(self) -> str:
        return str.replace(str(self),"XType.", "")

    def do_analyze(self) -> bool:
        return True

    @staticmethod
    def parse(dtype: str):
        if dtype == "f32":
            return XType.f32
        elif dtype == "f16":
            return XType.f16
        elif dtype == "bf16":
            return XType.bf16
        elif dtype == "f8_e4m3":
            return XType.f8_e4m3
        elif dtype == "f8_e5m2":
            return XType.f8_e5m2
        elif dtype == "f4_e2m1":
            return XType.f4_e2m1
        elif dtype == "qi3":
            return XType.qi3
        elif dtype == "qi4":
            return XType.qi4
        elif dtype == "qi8":
            return XType.qi8

        elif dtype == "q4_0":
            return XType.q4_0
        elif dtype == "q4_1":
            return XType.q4_1
        elif dtype == "q5_0":
            return XType.q5_0
        elif dtype == "q5_1":
            return XType.q5_1
        elif dtype == "q8_0":
            return XType.q8_0
        elif dtype == "tq1_0":
            return XType.tq1_0
        else:
            raise ValueError(f"Unknown dtype {dtype}")

    @staticmethod
    def convert_to(t_from: torch.tensor, type_to, type_from=None) -> torch.tensor:
        t: torch.tensor = None

        if type_from is None:
            if t_from.dtype == torch.float32:
                t = t_from
            elif t_from.dtype == torch.bfloat16:
                t = t_from
            else:
                raise ValueError(f"Unsupported source type: {t_from.dtype}")
        else:
            if type_from == XType.f4_e2m1:
                t = dequantize_from_f4_e2m1(t_from)
            elif type_from == XType.f32:
                t = t_from
            elif type_from == XType.f16:
                t = t_from
            elif type_from == XType.bf16:
                t = t_from
            elif type_from == XType.f8_e4m3:
                t = t_from.to(torch.float32)
            elif type_from == XType.f8_e5m2:
                t = t_from.to(torch.float32)
            elif type_from == XType.qi8:
                t = t_from.to(dequantize_from_qi8(t_from))

            elif type_from == XType.q4_0:
                t = torch.from_numpy(gguf_dequantize(t_from.numpy(), GGMLQuantizationType.Q4_0))
            elif type_from == XType.q4_1:
                t = torch.from_numpy(gguf_dequantize(t_from.numpy(), GGMLQuantizationType.Q4_1))
            elif type_from == XType.q5_0:
                t = torch.from_numpy(gguf_dequantize(t_from.numpy(), GGMLQuantizationType.Q5_0))
            elif type_from == XType.q5_1:
                t = torch.from_numpy(gguf_dequantize(t_from.numpy(), GGMLQuantizationType.Q5_1))
            elif type_from == XType.q8_0:
                t = torch.from_numpy(gguf_dequantize(t_from.numpy(), GGMLQuantizationType.Q8_0))
            elif type_from == XType.tq1_0:
                t = torch.from_numpy(gguf_dequantize(t_from.numpy(), GGMLQuantizationType.TQ1_0))
            else:
                raise ValueError(f"Unsupported source type: {t_from.dtype}")

        if not t.dtype in [torch.float32, torch.float16, torch.bfloat16]:
            raise ValueError(f"Unsupported source type: {t.dtype}")

        if type_to == XType.f32:
            converted_tensor = t.to(torch.float32)
        elif type_to == XType.f16:
            converted_tensor = t.to(torch.float16)
        elif type_to == XType.bf16:
            converted_tensor = t.to(torch.bfloat16)
        elif type_to == XType.f8_e4m3:
            torchtype = getattr(torch, "float8_e4m3fn")
            converted_tensor = t.to(torchtype)
        elif type_to == XType.f8_e5m2:
            torchtype = getattr(torch, "float8_e5m2")
            converted_tensor = t.to(torchtype)
        elif type_to == XType.f4_e2m1:
            converted_tensor = quantize_to_f4_e2m1(t.to(torch.float32))
        elif type_to == XType.qi8:
            converted_tensor = quantize_to_qi8(t.to(torch.float32))
        elif type_to == XType.qi3:
            converted_tensor = convert_to_3bit(t.to(torch.float32))
        elif type_to == XType.qi4:
            converted_tensor = convert_to_4bit(t.to(torch.float32))
        elif type_to == XType.q4_0:
            converted_tensor = torch.from_numpy(gguf_quantize(t.to(torch.float32).numpy(), GGMLQuantizationType.Q4_0))
        elif type_to == XType.q4_1:
            converted_tensor = torch.from_numpy(gguf_quantize(t.to(torch.float32).numpy(), GGMLQuantizationType.Q4_1))
        elif type_to == XType.q5_0:
            converted_tensor = torch.from_numpy(gguf_quantize(t.to(torch.float32).numpy(), GGMLQuantizationType.Q5_0))
        elif type_to == XType.q5_1:
            converted_tensor = torch.from_numpy(gguf_quantize(t.to(torch.float32).numpy(), GGMLQuantizationType.Q5_1))
        elif type_to == XType.q8_0:
            converted_tensor = torch.from_numpy(gguf_quantize(t.to(torch.float32).numpy(), GGMLQuantizationType.Q8_0))
        elif type_to == XType.tq1_0:
            converted_tensor = torch.from_numpy(gguf_quantize(t.to(torch.float32).numpy(), GGMLQuantizationType.TQ1_0))
        else:
            raise ValueError(f"Unsupported target type: {type_to}")

        return converted_tensor

class Metadata:
    def __init__(self, config):
        arch = config["architectures"][0]
        if arch not in SUPPORTED_ARCHITECTURES:
            raise Exception(f"Architecture {arch} is not supported, must be one of {SUPPORTED_ARCHITECTURES}")
        self.arch = arch
        if arch == "MistralForCausalLM" or arch == "LlamaForCausalLM":
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
            self.tie_word_embeddings = config["tie_word_embeddings"]

            assert config["hidden_act"] in ["gelu", "silu"]
            self.act_type = config["hidden_act"]

            self.tensors = OrderedDict()
        else:
            raise Exception(f"unexpected Architecture: {arch}!")

    def to_dict(self):
        result = {"arch": self.arch}
        if self.arch == "MistralForCausalLM" or self.arch == "LlamaForCausalLM":
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
            result["tie_word_embeddings"] = str(self.tie_word_embeddings)


        result["tensors"] = str(self.tensors)

        #if len(str(result)) % 32 != 0:
        #    result["padding"] = ""
        #    while len(str(result)) % 32 != 0:
        # #       result["padding"] += chr(64 + (32 - (len(str(result)) % 32)))
        #return result

# this is a horrible gpt-2 unicode byte encoder hack from https://github.com/openai/gpt-2/blob/master/src/encoder.py#L9
# this has poisoned all HF tokenizer configs that use ByteLevel decoder/preprocessor
# as a result we get crazy UTF-8-as-bytes-as-UTF8 in the tokenizer data that we need to convert back
def gpt2_bytes_to_unicode():
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def load_tokens(tokenizer_path, vocab_size):
    print(f"Loading tokenizer from {tokenizer_path}")
    tokens = [""] * vocab_size
    with open(tokenizer_path, "r") as f:
        tokenizer = json.load(f)
    use_gpt2_byte_preprocessing = not tokenizer["model"].get("byte_fallback", False)

    vocab = tokenizer["model"]["vocab"]
    assert len(vocab) <= vocab_size

    for t, i in vocab.items():
        tokens[i] = t

    for added in tokenizer["added_tokens"]:
        tokens[added["id"]] = added["content"]

    gpt2_decode = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    # Preprocess tokens into UTF-8 encoding
    for i, t in enumerate(tokens):
        if use_gpt2_byte_preprocessing:
            b = bytes([gpt2_decode.get(c, 0) for c in t])
        else:
            t = t.replace('\u2581', ' ') # sentencepiece uses this character as whitespace
            b = t.encode('utf-8')
        b = b.replace(b"\0", b"\7") # replace null bytes with bell characters
        assert b.count(0) == 0 # no null bytes allowed
        tokens[i] = b

    return tokens

def convert_to_3bit(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert a tensor to 3-bit representation, packing weights efficiently into bytes.

    Assumes the input tensor has values in a known range (e.g., -1 to 1).
    The tensor size must be a multiple of 8 for optimal packing.

    Parameters:
        tensor (torch.Tensor): Input tensor (assumed float32).

    Returns:
        torch.Tensor: Packed tensor in 3-bit format (torch.uint8).
    """
    assert tensor.dtype == torch.float32, "Input tensor must be float32"

    # Normalize values from [-1, 1] to [0, 7] for 3-bit storage
    tensor = torch.clamp(tensor, -1, 1)  # Ensure values are within range
    tensor = ((tensor + 1) * 3.5).round().to(torch.uint8)  # Scale to 3-bit range (0-7)

    # Ensure tensor length is a multiple of 8 (since we store 8 values in 3 bytes)
    assert tensor.numel() % 8 == 0, "Tensor size must be a multiple of 8 for efficient 3-bit packing"

    # Packing: 8 values (3 bits each) fit into 3 bytes
    packed_tensor = (
        (tensor[::8] << 5) | (tensor[1::8] << 2) | (tensor[2::8] >> 1),  # First byte
        ((tensor[2::8] & 0x01) << 7) | (tensor[3::8] << 4) | (tensor[4::8] << 1) | (tensor[5::8] >> 2),  # Second byte
        ((tensor[5::8] & 0x03) << 6) | (tensor[6::8] << 3) | tensor[7::8],  # Third byte
    )

    # Concatenate into a single packed tensor
    packed_tensor = torch.stack(packed_tensor, dim=-1).flatten()

    return packed_tensor

def convert_to_4bit(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert a tensor to 4-bit representation, packing two values per byte.

    Assumes the input tensor has values in a known range (e.g., -1 to 1)
    and dimensions are multiples of 2.

    Parameters:
        tensor (torch.Tensor): Input tensor (assumed float32).

    Returns:
        torch.Tensor: Packed tensor in 4-bit format (torch.uint8).
    """
    assert tensor.dtype == torch.float32, "Input tensor must be float32"

    # Normalize values from [-1, 1] to [0, 15] for 4-bit storage
    tensor = torch.clamp(tensor, -1, 1)  # Ensure values are within range
    tensor = ((tensor + 1) * 7.5).round().to(torch.uint8)  # Scale to 4-bit range (0-15)

    # Pack two 4-bit values into a single 8-bit byte
    packed_tensor = (tensor[::2] << 4) | tensor[1::2]  # Shift first value and merge

    return packed_tensor


def pack_tensor(tensor: torch.tensor, bits: int, overflow_saturate: bool = False) -> torch.tensor:
    # pack source tensor tightly
    if tensor.dtype == torch.int8:
        if torch.any(tensor < 0):
            raise ValueError(f"Integer tensor {tensor} is negative")
        t = tensor.to(torch.uint8)
    elif tensor.dtype == torch.uint8:
        t = tensor
    elif tensor.dtype == torch.int32:
        if torch.any(tensor < 0):
            raise ValueError(f"Integer tensor {tensor} is negative")
        t = tensor.to(torch.uint8)
    else:
        raise ValueError(f"Input tensor must be of dtype uint8 or int8, but is {tensor.dtype}")

    max_value = (1 << bits) - 1
    if overflow_saturate:
        t = torch.clamp(t, 0, max_value)
    else:
        if torch.any(t > max_value):
            raise ValueError(f"Tensor contains values exceeding the maximum for {bits}-bit packing: {max_value}")

    if bits == 2:
        # Group 4 values into 1 byte (2 bits each)
        packed = torch.zeros((t.numel() + 3) // 4, dtype=torch.uint8)
        for i in range(0, t.numel(), 4):
            chunk = t[i:i+4].to(torch.uint8)
            chunk = torch.cat([chunk, torch.zeros(4 - len(chunk), dtype=torch.uint8)])  # Padding if needed
            packed[i // 4] = (chunk[0] << 6) | (chunk[1] << 4) | (chunk[2] << 2) | chunk[3]
        return packed

    if bits == 3:
        # Group 8 values into 3 bytes (3 bits each)
        packed = torch.zeros((t.numel() + 7) // 8 * 3, dtype=torch.uint8)
        for i in range(0, t.numel(), 8):
            chunk = t[i:i+8].to(torch.uint8)
            chunk = torch.cat([chunk, torch.zeros(8 - len(chunk), dtype=torch.uint8)])  # Padding if needed
            packed[i // 8 * 3 + 0] = (chunk[0] << 5) | (chunk[1] << 2) | (chunk[2] >> 1)
            packed[i // 8 * 3 + 1] = ((chunk[2] & 0b1) << 7) | (chunk[3] << 4) | (chunk[4] << 1) | (chunk[5] >> 2)
            packed[i // 8 * 3 + 2] = ((chunk[5] & 0b11) << 6) | (chunk[6] << 3) | chunk[7]
        return packed

    if bits == 4:
        packed = (t[::2] << 4) | t[1::2]
        return packed.to(torch.uint8)

    if bits == 5:
        # Group 8 values into 5 bytes (5 bits each)
        packed = torch.zeros((t.numel() + 7) // 8 * 5, dtype=torch.uint8)
        for i in range(0, t.numel(), 8):
            chunk = t[i:i+8].to(torch.uint8)
            chunk = torch.cat([chunk, torch.zeros(8 - len(chunk), dtype=torch.uint8)])  # Padding if needed
            packed[i // 8 * 5 + 0] = (chunk[0] << 3) | (chunk[1] >> 2)
            packed[i // 8 * 5 + 1] = ((chunk[1] & 0b11) << 6) | (chunk[2] << 1) | (chunk[3] >> 4)
            packed[i // 8 * 5 + 2] = ((chunk[3] & 0b1111) << 4) | (chunk[4] >> 1)
            packed[i // 8 * 5 + 3] = ((chunk[4] & 0b1) << 7) | (chunk[5] << 2) | (chunk[6] >> 3)
            packed[i // 8 * 5 + 4] = ((chunk[6] & 0b111) << 5) | chunk[7]
        return packed

    if bits == 6:
        # Group 4 values into 3 bytes (6 bits each)
        packed = torch.zeros((t.numel() + 3) // 4 * 3, dtype=torch.uint8)
        for i in range(0, t.numel(), 4):
            chunk = t[i:i+4].to(torch.uint8)
            chunk = torch.cat([chunk, torch.zeros(4 - len(chunk), dtype=torch.uint8)])  # Padding if needed
            packed[i // 4 * 3 + 0] = (chunk[0] << 2) | (chunk[1] >> 4)
            packed[i // 4 * 3 + 1] = ((chunk[1] & 0b1111) << 4) | (chunk[2] >> 2)
            packed[i // 4 * 3 + 2] = ((chunk[2] & 0b11) << 6) | chunk[3]
        return packed

    if bits == 7:
        # Group 8 values into 7 bytes (7 bits each)
        packed = torch.zeros((t.numel() + 7) // 8 * 7, dtype=torch.uint8)
        for i in range(0, t.numel(), 8):
            chunk = t[i:i+8].to(torch.uint8)
            chunk = torch.cat([chunk, torch.zeros(8 - len(chunk), dtype=torch.uint8)])  # Padding if needed
            packed[i // 8 * 7 + 0] = (chunk[0] << 1) | (chunk[1] >> 6)
            packed[i // 8 * 7 + 1] = ((chunk[1] & 0b111111) << 2) | (chunk[2] >> 5)
            packed[i // 8 * 7 + 2] = ((chunk[2] & 0b11111) << 3) | (chunk[3] >> 4)
            packed[i // 8 * 7 + 3] = ((chunk[3] & 0b1111) << 4) | (chunk[4] >> 3)
            packed[i // 8 * 7 + 4] = ((chunk[4] & 0b111) << 5) | (chunk[5] >> 2)
            packed[i // 8 * 7 + 5] = ((chunk[5] & 0b11) << 6) | (chunk[6] >> 1)
            packed[i // 8 * 7 + 6] = ((chunk[6] & 0b1) << 7) | chunk[7]
        return packed

    if bits == 10:
        # Group 4 values into 5 bytes (10 bits each)
        packed = torch.zeros((t.numel() + 3) // 4 * 5, dtype=torch.uint8)
        for i in range(0, t.numel(), 4):
            chunk = t[i:i+4].to(torch.uint8)
            chunk = torch.cat([chunk, torch.zeros(4 - len(chunk), dtype=torch.uint8)])  # Padding if needed
            packed[i // 4 * 5 + 0] = chunk[0]
            packed[i // 4 * 5 + 1] = (chunk[1] >> 2) | (chunk[0] << 8)
            packed[i // 4 * 5 + 2] = (chunk[1] << 6) | (chunk[2] >> 4)
            packed[i // 4 * 5 + 3] = (chunk[2] << 4) | (chunk[3] >> 6)
            packed[i // 4 * 5 + 4] = chunk[3] << 2
        return packed

    if bits == 12:
        # Group 2 values into 3 bytes (12 bits each)
        packed = torch.zeros((t.numel() + 1) // 2 * 3, dtype=torch.uint8)
        for i in range(0, t.numel(), 2):
            chunk = t[i:i+2].to(torch.uint8)
            chunk = torch.cat([chunk, torch.zeros(2 - len(chunk), dtype=torch.uint8)])  # Padding if needed
            packed[i // 2 * 3 + 0] = chunk[0]
            packed[i // 2 * 3 + 1] = (chunk[0] >> 4) | (chunk[1] << 4)
            packed[i // 2 * 3 + 2] = chunk[1] >> 8
        return packed

    raise ValueError(f"bits must be one of 2, 3, 4, 5, 6, 7, 10, 12!")

def quantize_to_qi8(tensor: torch.tensor) -> torch.tensor:
    # Map [-1, 1] -> [0, 255]
    x = torch.clamp(tensor, -1, 1)
    x = ((x + 1.0) * 127.5).round()
    q = x.clamp(0, 255).to(torch.uint8)
    return q

def dequantize_from_qi8(tensor: torch.tensor) -> torch.tensor:
    if tensor.dtype != torch.uint8:
        raise ValueError("Input tensor must be of dtype uint8")

    # Map [0, 255] -> [-1, 1]
    x = tensor.to(torch.float32)
    return (x / 127.5) - 1.0

def quantize_to_f4_e2m1(tensor: torch.tensor) -> torch.tensor:
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

    return pack_tensor(fp4, 4)

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

def quantize_to_f6_e3m2(tensor):
    if tensor.dtype != torch.float32:
        raise ValueError("Input tensor must be of dtype float32")

    # Extract sign
    sign_bit = (tensor < 0).to(torch.int32)
    abs_tensor = torch.abs(tensor)

    # Compute exponent (bias = 3) and clamp range
    exponent = torch.floor(torch.log2(abs_tensor)).clamp(-3, 4)
    exponent_bits = (exponent + 3).to(torch.int32)  # Add bias of 3

    # Normalize and compute mantissa
    normalized = abs_tensor / (2 ** exponent)
    mantissa = ((normalized - 1.0) * 4).clamp(0, 3).to(torch.int32)  # 2 bits for mantissa

    # Combine to form FP6 (1 sign bit | 3 exponent bits | 2 mantissa bits)
    fp6 = (sign_bit << 5) | (exponent_bits << 2) | mantissa

    # Pack FP6 values into 3 bytes (4 values per group)
    # Group values in 4s
    num_values = fp6.numel()
    padded_size = (num_values + 3) // 4 * 4  # Ensure multiples of 4
    fp6_padded = torch.cat([fp6, torch.zeros(padded_size - num_values, dtype=torch.int32)])

    # Pack into bytes
    packed = torch.zeros(padded_size // 4 * 3, dtype=torch.uint8)
    print(f"packed shape: {packed.shape}, dtype: {packed.dtype}")
    print(f"fp6_padded shape: {fp6_padded.shape}, dtype: {fp6_padded.dtype}")

    fp6_padded_flat = fp6_padded.flatten()  # Flatten to 1D
    for i in range(0, fp6_padded.numel(), 4):
        packed[i // 4 * 3 + 0] = ((fp6_padded_flat[i + 0] << 2) | (fp6_padded_flat[i + 1] >> 4)).to(torch.uint8)
        packed[i // 4 * 3 + 1] = (((fp6_padded_flat[i + 1] & 0xF) << 4) | (fp6_padded_flat[i + 2] >> 2)).to(torch.uint8)
        packed[i // 4 * 3 + 2] = (((fp6_padded_flat[i + 2] & 0x3) << 6) | fp6_padded_flat[i + 3]).to(torch.uint8)
    return packed

def dequantize_from_f6_e3m2(packed_tensor):
    # Unpack 3 bytes into 4 FP6 values
    num_values = packed_tensor.numel() * 4 // 3
    unpacked = torch.zeros(num_values, dtype=torch.int32)
    for i in range(0, num_values, 4):
        unpacked[i + 0] = (packed_tensor[i // 4 * 3 + 0] >> 2) & 0x3F
        unpacked[i + 1] = ((packed_tensor[i // 4 * 3 + 0] & 0x3) << 4) | ((packed_tensor[i // 4 * 3 + 1] >> 4) & 0xF)
        unpacked[i + 2] = ((packed_tensor[i // 4 * 3 + 1] & 0xF) << 2) | ((packed_tensor[i // 4 * 3 + 2] >> 6) & 0x3)
        unpacked[i + 3] = packed_tensor[i // 4 * 3 + 2] & 0x3F

    # Extract components
    sign_bit = (unpacked >> 5) & 0b1
    exponent_bits = (unpacked >> 2) & 0b111
    mantissa_bits = unpacked & 0b11

    # Decode values
    sign = (-1) ** sign_bit
    exponent = exponent_bits - 3  # Remove bias of 3
    mantissa = mantissa_bits / 4
    value = sign * (1 + mantissa) * (2 ** exponent)
    return value.to(torch.float32)

def translate_name(name: str):
    if name == "model.embed_tokens.weight":
        return "embed.weight"

    if name == "model.norm.weight":
        return "output.norm.weight"

    if name == "lm_head.weight":
        return "output.weight"

    # model.layers.{l}.input_layernorm.weight -> model.layers.{l}.attn.norm.weight
    name = name.replace("model.layers.", "l.")
    # model.layers.{l}.self_attn.q_proj.weight
    name = name.replace(".self_attn.q_proj.", ".attn.q.")
    name = name.replace(".self_attn.k_proj.", ".attn.k.")
    name = name.replace(".self_attn.v_proj.", ".attn.v.")
    name = name.replace(".self_attn.o_proj.", ".attn.down.")
    # model.layers.{l}.post_attention_layernorm.weight
    name = name.replace(".post_attention_layernorm.", ".mlp.norm.")
    # model.layers.{l}.input_layernorm.weight
    name = name.replace(".input_layernorm.", ".attn.norm.")
    # model.layers.{l}.mlp.gate_proj.weight
    name = name.replace(".mlp.gate_proj.", ".mlp.gate.")
    # model.layers.{l}.mlp.down_proj.weight
    name = name.replace(".mlp.down_proj.", ".mlp.down.")
    # model.layers.{l}.mlp.up_proj.weight
    name = name.replace(".mlp.up_proj.", ".mlp.up.")

    return name

def fmt_number(value):
    """
    Converts a float or integer into a human-readable string with suffixes.
    """
    suffixes = ['','k', 'm', 'b', 't']  # Thousand, Million, Billion, Trillion
    magnitude = 0

    while abs(value) >= 1000 and magnitude < len(suffixes) - 1:
        magnitude += 1
        value /= 1000.0

    return f"{value:.1f}{suffixes[magnitude]}"

def load_weights(model_files, target_type: XType, metadata, tie_word_embeddings) -> dict[str, torch.tensor]:
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

    def boost_type(org_type: XType, cvt_type: XType) -> XType:
        if org_type == XType.bf16:
            if cvt_type == XType.f8_e4m3 or cvt_type == XType.f8_e4m3 or cvt_type == XType.qi8:
                return XType.bf16
            if cvt_type == XType.qi3 or cvt_type == XType.qi4:
                return XType.f8_e5m2
            return cvt_type

        if org_type == XType.f32 or org_type == XType.f16:
            if cvt_type == XType.f8_e4m3 or cvt_type == XType.f8_e4m3 or cvt_type == XType.qi8:
                return XType.f16
            if cvt_type == XType.qi3 or cvt_type == XType.qi4:
                return XType.f8_e4m3
            return cvt_type

        return cvt_type

    # convert weights
    progress = 0
    def conv(name: str):
        nonlocal progress

        t: torch.tensor = weights[name]
        conv_name: str = translate_name(name)

        if conv_name.find("attn.q.weight") > 0:
            t = permute_reverse(t, n_heads, rotary_dim)

        if conv_name.find("attn.k.weight") > 0:
            t = permute_reverse(t, n_kv_heads, rotary_dim)

        progress += 1
        actual_type = target_type
        if conv_name == "embed.weight" or conv_name == "output.weight":
            if t.dtype == torch.bfloat16:
                actual_type = boost_type(XType.bf16, target_type)
            elif t.dtype == torch.float16:
                actual_type = boost_type(XType.f16, target_type)
            elif t.dtype == torch.float32:
                actual_type = boost_type(XType.f32, target_type)

        if len(t.shape) == 1:
            if t.dtype == torch.bfloat16:
                actual_type = XType.bf16
            else:
                actual_type = XType.f32

        if args.analyze:
            v_range = torch.max(t) - torch.min(t)
            r_range = 16.0 / (torch.max(t, 0).values - torch.min(t, 0).values)
            print(f"{conv_name}[{t.dtype}, range={v_range:.4f}]")

            for test_type in XType:
                if test_type == actual_type or name == "embed.weight":
                    continue
                if not test_type.do_analyze():
                    continue

                if test_type == XType.f16 or test_type == XType.f32 or test_type == XType.bf16:
                    scales = [1.0]
                else:
                    scales = [-1, 1.0, 8.0, 16.0, 1 / v_range * 4]

                for scale in scales:
                    o = t.to(torch.float32)

                    if scale == -1:
                        convt = XType.convert_to(t * r_range, test_type)
                        q = XType.convert_to(convt, XType.f32, type_from=test_type) / r_range
                    else:
                        convt = XType.convert_to(t * scale, test_type)
                        q = XType.convert_to(convt, XType.f32, type_from=test_type) / scale

                    mse = torch.mean((o * 1000 - q * 1000) ** 2)
                    normalized_error = torch.sum(torch.abs(o - q)) / torch.sum(torch.abs(o))
                    cos_sim = torch.cosine_similarity(o.view(-1).unsqueeze(0), q.view(-1).unsqueeze(0)).item()
                    snr = 10 * torch.log10(torch.sum(o ** 2) / torch.sum((o - q) ** 2))
                    accuracy = torch.mean((torch.abs(o - q) <= 0.0001).float()).item()
                    size_in_bytes = convt.element_size() * convt.nelement()

                    print(f"=> {test_type.name():<12}scale={scale:<8.2f}size={fmt_number(size_in_bytes):<10}mse={mse:<8.2f}norm_err_k={normalized_error:<8.2f}cosine={cos_sim:<8.3f}snr={snr:<8.2f}accuracy={accuracy:<8.2f}")
                print()
            print()
        else:
            print(f"{name:<50}{str(t.dtype).replace("torch.", ""):<8} => {conv_name:<24}{actual_type:<12}")

            convt: torch.tensor = XType.convert_to(t, actual_type)
            tensors[conv_name] = convt

            metadata.tensors.append({
                "type": actual_type.name(),
                "shape": convt.shape,
            })

    tensors = {}

    conv("model.embed_tokens.weight")

    for l in range(config["num_hidden_layers"]):
        conv(f"model.layers.{l}.input_layernorm.weight")

        rotary_dim = metadata.rotary_dim
        n_heads = metadata.n_heads
        n_kv_heads = metadata.n_kv_heads

        conv(f"model.layers.{l}.self_attn.q_proj.weight")
        conv(f"model.layers.{l}.self_attn.k_proj.weight")
        conv(f"model.layers.{l}.self_attn.v_proj.weight")
        conv(f"model.layers.{l}.self_attn.o_proj.weight")
        conv(f"model.layers.{l}.post_attention_layernorm.weight")

        conv(f"model.layers.{l}.mlp.gate_proj.weight")
        conv(f"model.layers.{l}.mlp.down_proj.weight")
        conv(f"model.layers.{l}.mlp.up_proj.weight")

        gc.collect()

    if not tie_word_embeddings:
        conv("lm_head.weight")
    conv("model.norm.weight")

    gc.collect()

    return tensors

def download_file(url: str, output_path: str, token: str = None, overwrite: bool = False) -> bool:
    """
    Download a file from the given URL and save it to the specified path with a progress bar.
    """

    if os.path.exists(output_path):
        if overwrite:
            os.remove(output_path)
        else:
            print(f"downloading {url}...{output_path} already exists, skipping.")
            return True

    print(f"downloading {url}...")

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
        return True
    else:
        print(f"Failed to download {url}. HTTP Status: {response.status_code}")
        return False

def process_input(input_path: str) -> (str, str, []):
    model_files = {}

    config_file = None
    tokenizer_file = None
    model_files = []

    if os.path.exists(os.path.join(input_path, "config.json")):
        config_file = os.path.join(input_path, "config.json")
        print(f"using {config_file}\n")
    else:
        raise FileNotFoundError("config.json not found")

    if os.path.exists(os.path.join(input_path, "tokenizer.json")):
        tokenizer_file = os.path.join(input_path, "tokenizer.json")
        print(f"using {tokenizer_file}\n")
    else:
        raise FileNotFoundError("tokenizer.json not found")

    if os.path.exists(os.path.join(input_path, "model.safetensors")):
        model_files.append(os.path.join(input_path, "model.safetensors"))
        print(f"using {model_files}\n")
    else:
        if os.path.exists(os.path.join(input_path, "model-00001-of-00003.safetensors")):
            files_to_process = [
                "model-00001-of-00003.safetensors",
                "model-00002-of-00003.safetensors",
                "model-00003-of-00003.safetensors"
            ]
            for st_f in files_to_process:
                if os.path.exists(os.path.join(input_path, st_f)):
                    model_files.append(os.path.join(input_path, st_f))
        elif os.path.exists(os.path.join(input_path, "model-00001-of-00004.safetensors")):
            files_to_process = [
                "model-00001-of-00004.safetensors",
                "model-00002-of-00004.safetensors",
                "model-00003-of-00004.safetensors",
                "model-00004-of-00004.safetensors"
            ]
            for st_f in files_to_process:
                if os.path.exists(os.path.join(input_path, st_f)):
                    model_files.append(os.path.join(input_path, st_f))
        elif os.path.exists(os.path.join(input_path, "model-00001-of-00010.safetensors")):
            files_to_process = [
                "model-00001-of-00010.safetensors",
                "model-00002-of-00010.safetensors",
                "model-00003-of-00010.safetensors",
                "model-00004-of-00010.safetensors",
                "model-00005-of-00010.safetensors",
                "model-00006-of-00010.safetensors",
                "model-00007-of-00010.safetensors",
                "model-00008-of-00010.safetensors",
                "model-00009-of-00010.safetensors",
                "model-00010-of-00010.safetensors"
            ]
            for st_f in files_to_process:
                if os.path.exists(os.path.join(input_path, st_f)):
                    model_files.append(os.path.join(input_path, st_f))

        print(f"using {model_files}")
        if len(model_files) == 0:
            raise FileNotFoundError("no model files found!")

    return config_file, tokenizer_file, model_files


def download_model(url: str, token: str = None) -> dict:
    # Download files from URL
    # https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/raw/main/tokenizer.json
    # https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/resolve/main/model-00001-of-00003.safetensors

    downloaded_files = {}
    if not(url.startswith("http://") or url.startswith("https://")):
        raise Exception(f"Invalid URL: {url}")

    print(f"Download model from: {url}...\n")
    model_name = url.strip("/").split("/")[-1]
    print(f"Model name: {model_name}...\n")

    temp_dir = os.path.join("./", model_name)

    os.makedirs(temp_dir, exist_ok=True)

    # model config
    config_file = "config.json"
    config_url = urljoin(url + "raw/main/", config_file)
    config_path = os.path.join(temp_dir, config_file)

    if download_file(config_url, config_path, token):
        downloaded_files[config_file] = config_path
    else:
        raise Exception(f"Failed to download {config_url}.\n")

    # tokenizer
    tokenizer_file = "tokenizer.json"
    tokenizer_url = urljoin(url + "resolve/main/", tokenizer_file)
    tokenizer_path = os.path.join(temp_dir, tokenizer_file)

    if download_file(tokenizer_url, tokenizer_path, token):
        downloaded_files[tokenizer_file] = tokenizer_path
    else:
        raise Exception(f"Failed to download {tokenizer_url}.\n")

    safetensors_file = "model.safetensors"
    safetensors_url = urljoin(url + "resolve/main/", safetensors_file)
    safetensors_path = os.path.join(temp_dir, safetensors_file)

    if download_file(safetensors_url, safetensors_path, token):
        downloaded_files[safetensors_file] = safetensors_path
        return downloaded_files

    # 3 parts
    safetensors_file = "model-00001-of-00003.safetensors"
    safetensors_url = urljoin(url + "resolve/main/", safetensors_file)
    safetensors_path = os.path.join(temp_dir, safetensors_file)

    if download_file(safetensors_url, safetensors_path, token):
        downloaded_files[safetensors_file] = safetensors_path
        for safetensors_file in ["model-00002-of-00003.safetensors", "model-00003-of-00003.safetensors"]:
            safetensors_url = urljoin(url + "resolve/main/", safetensors_path)
            safetensors_path = os.path.join(temp_dir, safetensors_path)
            if download_file(safetensors_url, safetensors_path, token):
                downloaded_files[safetensors_file] = safetensors_path
            else:
                raise Exception(f"Failed to download {safetensors_file}.")
        return downloaded_files

    # 4 parts
    safetensors_file = "model-00001-of-00004.safetensors"
    safetensors_url = urljoin(url + "resolve/main/", safetensors_file)
    safetensors_path = os.path.join(temp_dir, safetensors_file)

    if download_file(safetensors_url, safetensors_path, token):
        downloaded_files[safetensors_file] = safetensors_path
        for safetensors_file in ["model-00002-of-00004.safetensors", "model-00003-of-00004.safetensors",
                                 "model-00004-of-00004.safetensors"]:
            safetensors_url = urljoin(url + "resolve/main/", safetensors_file)
            safetensors_path = os.path.join(temp_dir, safetensors_file)
            if download_file(safetensors_url, safetensors_path, token):
                downloaded_files[safetensors_file] = safetensors_path
            else:
                raise Exception(f"Failed to download {safetensors_file}.")
        return downloaded_files

    # 10 parts
    safetensors_file = "model-00001-of-00010.safetensors"
    safetensors_url = urljoin(url + "resolve/main/", safetensors_file)
    safetensors_path = os.path.join(temp_dir, safetensors_file)

    if download_file(safetensors_url, safetensors_path, token):
        downloaded_files[safetensors_file] = safetensors_path
        for safetensors_file in ["model-00002-of-00010.safetensors", "model-00003-of-00010.safetensors",
                                 "model-00004-of-00010.safetensors", "model-00005-of-00010.safetensors",
                                 "model-00006-of-00010.safetensors", "model-00007-of-00010.safetensors",
                                 "model-00008-of-00010.safetensors", "model-00009-of-00010.safetensors",
                                 "model-00010-of-00010.safetensors"]:
            safetensors_url = urljoin(url + "resolve/main/", safetensors_file)
            safetensors_path = os.path.join(temp_dir, safetensors_file)
            if download_file(safetensors_url, safetensors_path, token):
                downloaded_files[safetensors_file] = safetensors_path
            else:
                raise Exception(f"Failed to download {safetensors_file}.")
        return downloaded_files

def sort_tensors(tensors: dict) -> OrderedDict:
    """
    Sorts the tensors according to the specified order and returns an OrderedDict.
    """
    ordered_keys = []
    layer_keys = {}

    # Regex pattern to extract layer numbers
    layer_pattern = re.compile(r"l\.(\d+)\.")

    for key in tensors.keys():
        if key == "embed.weight":
            ordered_keys.insert(0, key)  # Embed weight first
        elif key == "output.weight":
            continue  # Handle this separately to place it before output.norm.weight
        elif key == "output.norm.weight":
            continue  # Handle this separately to place it before tokenizer.tokens
        elif key == "tokenizer.tokens":
            continue  # Handle this separately to place it last
        else:
            match = layer_pattern.search(key)
            assert match
            layer_num = int(match.group(1))
            if layer_num not in layer_keys:
                layer_keys[layer_num] = []
            layer_keys[layer_num].append(key)

    # Sorting within each layer
    layer_order = [
        "attn.norm.weight",
        "mlp.norm.weight",
        "attn.q.weight",
        "attn.k.weight",
        "attn.v.weight",
        "attn.down.weight",
        "mlp.gate.weight",
        "mlp.down.weight",
        "mlp.up.weight"
    ]

    for layer_num in sorted(layer_keys.keys(), key=int):  # Ensure numeric sorting
        sorted_layer = sorted(layer_keys[layer_num], key=lambda k: layer_order.index(k.split(f"l.{layer_num}.")[-1]) if k.split(f"l.{layer_num}.")[-1] in layer_order else 100)
        ordered_keys.extend(sorted_layer)

    if "output.weight" in tensors:
        ordered_keys.append("output.weight")
    if "output.norm.weight" in tensors:
        ordered_keys.append("output.norm.weight")
    if "tokenizer.tokens" in tensors:
        ordered_keys.append("tokenizer.tokens")

    return OrderedDict((key, tensors[key]) for key in ordered_keys)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert model files with tokenizer and config, supporting URL input.")

    parser.add_argument("--download", type=str, help="Base URL to download from huggingface.", required=False)
    parser.add_argument("--input", type=str, help="Base URL/path with files.", required=False)
    parser.add_argument("--output", type=str, help="Output file path for the converted model.", required=False)
    parser.add_argument("--type", type=str, default="f16", choices=XType.get_supported_types())
    parser.add_argument("--token", type=str, default=os.environ.get("HF_TOKEN"), help="Hugging Face access token for private models.", required=False)
    parser.add_argument("--analyze", type=bool, nargs="?", const=True, help="Only analyze the input file.", default=False)

    args = parser.parse_args()

    if not args.download is None:
        files = download_model(args.download, args.token)
        print(f"Downloaded {files.values()}")
        exit(0)

    print(f"torch version: {torch.__version__}")

    if not args.input is None:
        if args.output is None or args.output == "":
            args.output = os.path.join("./", args.input.strip("/").split("/")[-1] + f".{args.type}.xalm")
        config_file, tokenizer_file, model_files = process_input(args.input)

        with open(config_file, "r") as f:
            config = json.load(f)
            metadata = Metadata(config)

        print()
        tokens = load_tokens(tokenizer_file, metadata.vocab_size)
        print()
        tensors = load_weights(model_files, XType.parse(args.type), metadata, config.get("tie_word_embeddings", None))

        # add tokenizer tensors at the end (to maximize the chance of model tensor alignment)
        # note: we concatenate all bytes of all tokens into a single tensor
        tensors["tokenizer.tokens"] = torch.cat([torch.tensor([x for x in b] + [0], dtype=torch.uint8) for b in tokens])

        if not args.analyze:
            print(f"Saving {len(tensors)} tensors...")
            for k, v in tensors.items():
                assert v.layout == torch.strided and v.is_contiguous()

            gc.collect()
            save_file(sort_tensors(tensors), args.output, metadata.to_dict())
            print(sort_tensors(tensors).keys())

            print(f"saved to {args.output}")

        sys.exit(0)

    print("no task!")
    sys.exit(1)