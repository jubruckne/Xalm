#include "tokenizer.h"
#include <sstream>

static std::vector<int> parse_str(const std::string& input) {
	std::vector<int> result;

	if (!input.empty() && input.front() == '[' && input.back() == ']') {
		// Remove brackets
		std::string trimmed = input.substr(1, input.size() - 2);

		// Split by commas
		std::stringstream ss(trimmed);
		std::string token;
		while (std::getline(ss, token, ',')) {
			result.push_back(std::stoi(token));
		}
	} else {
		result.push_back(std::stoi(input));
	}
	return result;
}

Tokenizer::Tokenizer(const Xalm::file_info& data) {
	auto bos = data.metadata.at("bos_token_id").get<std::string>();
	auto eos = data.metadata.at("eos_token_id").get<std::string>();

	this->bos_id = parse_str(bos)[0];
	this->eos_id = parse_str(eos)[0];

	// TODO: figure out edge cases:
	// Q: should `vocab` include byte fallback tokens?
	// Q: should `vocab` include special tokens, e.g. '<unk>', '<s>', '</s>'?
	auto tokens_tensor = data.load_tensor("tokenizer.tokens");
	assert(tokens_tensor.type == Type::U8);

	const char* tokens_tensor_end = tokens_tensor.get_buffer()->get<char>() + tokens_tensor.linear_length;
	for (auto ptr = tokens_tensor.get_buffer()->get<char>(); ptr < tokens_tensor_end; ptr++) {
		char* s = ptr;
		while (*ptr != '\0' && ptr < tokens_tensor_end) {
			ptr++;
		}
		vocab.emplace_back(s, ptr - s);
	}
	for (size_t i = 0; i < vocab.size(); i++) {
		if (vocab[i] == "<0x00>") {
			byte_fallback_start = i;
		} else if (vocab[i] == "<|eot_id|>" || vocab[i] == "<|end|>" || vocab[i] == "<|im_end|>") {
			eot_id = i;
		}
	}
	// init byte_pieces
	for (size_t i = 0; i < 256; i++) {
		byte_pieces[i] = (char) i;
	}
	// init vocab trie
	for (size_t i = 0; i < vocab.size(); i++) {
		const std::string& word = vocab[i];
		TokenTrie* p = &vocab_trie;
		for (char c: word) {
			if (p->children.count(c) == 0) {
				p->children[c] = std::make_shared<TokenTrie>();
			}
			p = p->children[c].get();
		}
		p->token_id = i;
	}
}

std::string Tokenizer::decode_one(const int prev_token, const int token) const {
	const std::string& piece = vocab[token];
	// if following BOS token, sentencepiece decoder strips any leading whitespace
	if (prev_token == bos_id && piece[0] == ' ') {
		return piece.substr(1);
	}
	// return byte piece for byte fallback tokens (<0x00>, <0x01>, ..., <0xFF>)
	if (byte_fallback_start >= 0 && token >= byte_fallback_start && (token - byte_fallback_start) < 256) {
		return byte_pieces[token - byte_fallback_start];
	}
	return piece;
}

std::vector<int> Tokenizer::encode(const std::string& text, const bool encode_bos) const {
	std::vector<int> out_tokens;
	if (encode_bos) {
		out_tokens.push_back(bos_id);
	}

	for (size_t i = 0; i < text.size();) {
		size_t l = 0;
		size_t valid_l = 0;
		const TokenTrie* p = &vocab_trie;
		const TokenTrie* valid_p = nullptr;
		while (i + l < text.size()) {
			char c = text[i + l];
			if (p->children.count(c)) {
				p = p->children.at(c).get();
				l += 1;
				if (p->token_id >= 0) {
					valid_p = p;
					valid_l = l;
				}
			} else {
				break;
			}
		}
		if (!valid_p) {
			// No substring starting from `i` matches any vocab words, use byte fallback
			if (byte_fallback_start >= 0) {
				out_tokens.push_back((unsigned char) text[i] + byte_fallback_start);
			}
			i += 1;
		} else {
			out_tokens.push_back(valid_p->token_id);
			i += valid_l;
		}
	}

	return out_tokens;
}

std::string Tokenizer::encoding_to_debug_string(const std::vector<int>& encoding) const {
	std::string token_encoding_debug_str = "";
	for (const int token_id: encoding) {
		if (token_id == bos_id) {
			token_encoding_debug_str += "[<s>:" + std::to_string(token_id) + "]";
		} else if (token_id == eos_id) {
			token_encoding_debug_str += "[</s>:" + std::to_string(token_id) + "]";
		} else {
			token_encoding_debug_str += "[" + vocab[token_id] + ":" + std::to_string(token_id) + "]";
		}
	}
	return token_encoding_debug_str;
}
