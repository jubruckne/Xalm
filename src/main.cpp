#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

#include "fmt/format.h"
#include "profiler.h"

#include <cxxopts.hpp>
#include "model.h"
#include "sampler.h"
#include "tensor.h"
#include "time.h"
#include "tokenizer.h"

#include "console.h"


void error_usage() {
	fprintf(stderr, "Usage:   main <checkpoint> [options]\n");
	fprintf(stderr, "Example: main model.yalm -i \"Q: What is the meaning of life?\"\n");
	fprintf(stderr, "Options:\n");
	fprintf(stderr, "  -h Display this help message\n");
	fprintf(stderr, "  -d [cpu,cuda] which device to use (default - cuda)\n");
	fprintf(stderr, "  -m [completion,passkey,perplexity] which mode to run in (default - completion)\n");
	fprintf(stderr, "  -T <int> sliding window context length (0 - max)\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "Perplexity mode options:\n");
	fprintf(stderr, "  Choose one:\n");
	fprintf(stderr, "    -i <string> input prompt\n");
	fprintf(stderr, "    -f <filepath> input file with prompt\n");
	fprintf(stderr, "Completion mode options:\n");
	fprintf(stderr, "  -n <int>    number of steps to run for in completion mode, default 256. 0 = max_seq_len, -1 = "
					"infinite\n");
	fprintf(stderr, "  Choose one:\n");
	fprintf(stderr, "    -i <string> input prompt\n");
	fprintf(stderr, "    -f <filepath> input file with prompt\n");
	fprintf(stderr, "Passkey mode options:\n");
	fprintf(stderr, "  -n <int>    number of junk lines to insert (default - 250)\n");
	fprintf(stderr, "  -l <int>    passkey position (-1 - random)\n");
	exit(1);
}

#if DEBUG_MODEL
void debug_tensors(Config &c) {
	assert(debug_map_cpu().size() == debug_map_cuda().size());
	for (auto &[name, cpu]: debug_map_cpu()) {
		DebugTensor &cuda = debug_map_cuda().at(name);
		float maxerr = cpu.max_err(cuda);
		std::cout << fmt::format("{} maxerr: {}", name, maxerr) << std::endl;
	}
	std::cout << std::endl;
}
#endif

void run_completion(const std::string& checkpoint_path, const std::string &device, const std::string &prompt,
					const int context, int num_steps) {
	auto model_data = Xalm::load(checkpoint_path);
	auto model = Model::from_xalm(model_data, context);

	console::print("{}", model_data.format());

	InferenceState state(model.config);
	Sampler sampler(model.config);
	Tokenizer tokenizer(model_data);

	console::print("Model active bytes with full context window: {}\n", model.active_bytes(model.config.max_seq_len));

	if (num_steps == 0) {
		// `-n 0` means use the full context length
		num_steps = model.config.max_seq_len;
	}
	if (device == "cuda") {
		std::cout << "Using CUDA" << std::endl;
		// model.cuda();
		// state.cuda();
	}

	// Do one inference as warmup.
	// On CPU, this ensures all tensors are loaded into memory via mmap.
	// On GPU, this ensures all tensors are loaded into device memory and
	// kernels are compiled + instantiated.
	model.forward(state, 0, 0);

	std::vector<int> encoding;
	{
		uint64_t encode_start_ms = get_timestamp_ms();
		encoding = tokenizer.encode(prompt, true);
		uint64_t encode_end_ms = get_timestamp_ms();

		std::cout << tokenizer.encoding_to_debug_string(encoding) << std::endl;
		uint64_t encoding_ms = encode_end_ms - encode_start_ms;
		std::cout
				<< fmt::format(
						   "Encoding stats: ({} tokens, throughput: {:.5}tok/s, latency: {:.5}s/tok, total: {:.5}s)\n",
						   encoding.size(), encoding.size() / (encoding_ms / 1000.0),
						   (encoding_ms / 1000.0) / encoding.size(), encoding_ms / 1000.0)
				<< std::endl;
	}

	uint64_t start_ms = get_timestamp_ms();
	size_t read_bytes = 0;
	// Hydrate KV cache by forwarding model on all prompt tokens and discarding output.
	// This also generates output logits for the last token.
	for (size_t pos = 0; pos < encoding.size(); pos++) {
		int token_id = encoding[pos];
		InferenceMode inferMode =
				pos + 1 == encoding.size() ? InferenceMode::OUTPUT_LOGITS : InferenceMode::HYDRATE_KV_CACHE;
		model.forward(state, token_id, pos, inferMode);
		read_bytes += model.active_bytes(pos);
	}
	uint64_t end_hydrate_ms = get_timestamp_ms();
	// For N steps:
	// - Sample + decode output logits
	// - Forward the model
	for (int i = 0; i < num_steps || num_steps == -1; i++) {
		int token_id = sampler.sample_argmax(state);
		std::string token_str = tokenizer.decode_one(encoding.back(), token_id);
		std::cout << token_str << std::flush;
		encoding.push_back(token_id);
		if (token_id == tokenizer.eos_id || token_id == tokenizer.eot_id) {
			break;
		}
		model.forward(state, token_id, encoding.size() - 1);
		read_bytes += model.active_bytes(encoding.size() - 1);
	}
	std::cout << "\n" << std::endl;
	uint64_t end_ms = get_timestamp_ms();
	double elapsed_s = (end_ms - start_ms) / 1000.0;
	std::cout << fmt::format("Generation stats:\n"
							 "  {} tokens\n"
							 "  throughput: {:.5}tok/s\n"
							 "  latency: {:.5}s/tok\n"
							 "  hydrate: {:.5}s\n"
							 "  bandwidth: {:.5}GB/s\n"
							 "  total: {:.5}s\n",
							 encoding.size(), encoding.size() / elapsed_s, elapsed_s / encoding.size(),
							 (end_hydrate_ms - start_ms) / 1000.0, (static_cast<double>(read_bytes) / 1e9) / elapsed_s,
							 elapsed_s)
			  << std::endl;
}

void run_test(const std::string &checkpoint_path) {
	printf("test start\n");

	const auto model1 = Xalm::load("../models/Llama-3.2-1B-Instruct.f8_e4m3.xalm");


	std::print("{}", model1.format());


	auto t1 = model1.load_tensor("l.0.attn.down.weight");
	// Tensor t2 = model2.tensors.at("l.30.mlp.up.weight");
	std::print("\n{}", t1.format(8, 8));
	t1.save_to_csv("../models/l.0.attn.down.weight.txt");
	// std::print("\n{}", t2.format(8, 8));
	// std::print("\n{}", t1.format(16, 0, 8, 8));

	// Tensor ttt2 = model2.tensors.at("model.layers.0.attn.norm.weight");
	// std::printf("%s", ttt2.to_string(16).c_str());

	using flt = custom_float<2, 2, true, false, 3>;  // Linear mapping
	//using f8_log = custom_float<3, 4, 3, true, true, logarithmic_mapping<4>()>;
	//using f8_piecewise = custom_float<3, 3, 2, false, false, piecewise_linear_mapping<3>()>;

	//auto x1 = f8_standard::from(1.25f);

	console::print(flt::describe());;
	//auto x2 = f8_log::from(1.25f);
	//auto x3 = f8_piecewise::from(1.25f);

	return;

	auto a1 = Tensor::uniform(Type::F32, {4096 * 4, 4096}, 0, 1, "a");
	auto b0 = Tensor::uniform(Type::F32, {4096 * 4, 4096}, 0, 1, "b0");
	auto b1 = Tensor::uniform(Type::F32, {4096 * 4, 4096}, 0, 1, "b1");
	auto b2 = Tensor::uniform(Type::F8_E4M3, {4096 * 4, 4096}, 0, 1, "b2");
	auto out1 = Tensor::zeroes(Type::F32, {4096 * 4, 4096}, "out");

	printf("%s: size: %zu, length: %zu\n", a1.name.c_str(), a1.size, a1.linear_length);
	printf("%s: size: %zu, length: %zu\n", b0.name.c_str(), b0.size, b0.linear_length);
	printf("%s: size: %zu, length: %zu\n", b1.name.c_str(), b1.size, b1.linear_length);
	printf("%s: size: %zu, length: %zu\n", b2.name.c_str(), b2.size, b2.linear_length);
	printf("%s: size: %zu, length: %zu\n", out1.name.c_str(), out1.size, out1.linear_length);

	fflush(stdout);

	for (int i = 0; i < 256; i++) {
		matmul(out1, a1, b0);
		fflush(stdout);

		matmul(out1, a1, b1);
		fflush(stdout);

		matmul(out1, a1, b2);
		fflush(stdout);
	}
}

void run_perplexity(const std::string &checkpoint_path, const std::string &device, const std::string &prompt,
					const int context) {
	auto model_data = Xalm::load(checkpoint_path);
	auto model = Model::from_xalm(model_data, context);
	InferenceState state(model.config);
	Sampler sampler(model.config);
	Tokenizer tokenizer(model_data);

	std::cout << "Model active bytes with full context window: "
			  << model.active_bytes(model.config.max_seq_len) << std::endl;

	if (device == "cuda") {
		std::cout << "Using CUDA" << std::endl;
		// model.cuda();
		// state.cuda();
	}

	// Do one inference as warmup.
	// On CPU, this ensures all tensors are loaded into memory via mmap.
	// On GPU, this ensures all tensors are loaded into device memory and
	// kernels are compiled + instantiated.
	model.forward(state, 0, 0);

	std::vector<int> encoding;
	{
		uint64_t encode_start_ms = get_timestamp_ms();
		encoding = tokenizer.encode(prompt, true);
		uint64_t encode_end_ms = get_timestamp_ms();

		std::cout << tokenizer.encoding_to_debug_string(encoding) << std::endl;
		uint64_t encoding_ms = encode_end_ms - encode_start_ms;
		std::cout
				<< fmt::format(
						   "Encoding stats: ({} tokens, throughput: {:.5}tok/s, latency: {:.5}s/tok, total: {:.5}s)\n",
						   encoding.size(), encoding.size() / (encoding_ms / 1000.0),
						   (encoding_ms / 1000.0) / encoding.size(), encoding_ms / 1000.0)
				<< std::endl;
	}

	double sum_logprob = 0.0;
	double ss_logprob = 0.0;
	// Generates output logits for all tokens in the prompt and sum log probs to
	// compute perplexity.
	uint64_t start_ms = get_timestamp_ms();
	size_t read_bytes = 0;
	size_t N = encoding.size() - 1;
	for (size_t pos = 0; pos + 1 < encoding.size(); pos++) {
		std::cout << "\r Computing perplexity..." << pos + 1 << "/" << N << std::flush;

		int token_id = encoding[pos];
		model.forward(state, token_id, pos);
		read_bytes += model.active_bytes(pos);

		double logprob = std::log(sampler.sample_prob(encoding[pos + 1], state));
		sum_logprob += logprob;
		ss_logprob += logprob * logprob;
	}
	std::cout << std::endl;
	uint64_t end_ms = get_timestamp_ms();
	double elapsed_s = (end_ms - start_ms) / 1000.0;
	double perplexity = std::exp(-sum_logprob / N);
	double perplexity_error = perplexity * std::sqrt((ss_logprob - sum_logprob * sum_logprob / N) / N / N);
	std::cout << fmt::format("Stats:\n"
							 "  {} tokens\n"
							 "  perplexity: {:.5} ± {:.5}\n"
							 "  throughput: {:.5}tok/s\n"
							 "  latency: {:.5}s/tok\n"
							 "  bandwidth: {:.5}GB/s\n"
							 "  total: {:.5}s\n",
							 N, perplexity, perplexity_error, N / elapsed_s, elapsed_s / N,
							 (static_cast<double>(read_bytes) / 1e9) / elapsed_s, elapsed_s)
			  << std::endl;
}

void run_passkey(const std::string &checkpoint_path, const std::string &device, const int context, const int n_junk,
				 const int passkey_pos) {
	auto model_data = Xalm::load(checkpoint_path);
	auto model = Model::from_xalm(model_data, context);
	InferenceState state(model.config);
	Sampler sampler(model.config);
	Tokenizer tokenizer(model_data);

	std::cout << "Model active bytes with full context window: "
			  << model.active_bytes(model.config.max_seq_len) << std::endl;

	if (device == "cuda") {
		std::cout << "Using CUDA" << std::endl;
		// model.cuda();
		// state.cuda();
	}

	// Do one inference as warmup.
	// On CPU, this ensures all tensors are loaded into memory via mmap.
	// On GPU, this ensures all tensors are loaded into device memory and
	// kernels are compiled + instantiated.
	model.forward(state, 0, 0);

	const std::string PROMPT_PREFIX =
			"There is an important info hidden inside a lot of irrelevant text. "
			"Find it and memorize them. I will quiz you about the important information there.";
	const std::string PROMPT_SUFFIX = " What is the pass key? The pass key is";

	const int passkey = std::rand() % 50000 + 1;
	const int pos = passkey_pos == -1 ? std::rand() % n_junk : passkey_pos;

	std::string prompt = PROMPT_PREFIX;
	for (int i = 0; i < n_junk; i++) {
		if (i % n_junk == pos) {
			prompt += " The pass key is " + std::to_string(passkey) + ". Remember it. " + std::to_string(passkey) +
					  " is the pass key.";
		}
		prompt += " The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.";
	}
	prompt += PROMPT_SUFFIX;

	std::vector<int> encoding;
	{
		uint64_t encode_start_ms = get_timestamp_ms();
		encoding = tokenizer.encode(prompt, true);
		uint64_t encode_end_ms = get_timestamp_ms();

		uint64_t encoding_ms = encode_end_ms - encode_start_ms;
		std::cout
				<< fmt::format(
						   "Encoding stats: ({} tokens, throughput: {:.5}tok/s, latency: {:.5}s/tok, total: {:.5}s)\n",
						   encoding.size(), encoding.size() / (encoding_ms / 1000.0),
						   (encoding_ms / 1000.0) / encoding.size(), encoding_ms / 1000.0)
				<< std::endl;
	}

	// Allow max 16 steps to generate passkey
	constexpr size_t MAX_GENERATION_STEPS = 16;

	std::cout << fmt::format("Passkey test:\n"
							 "  prompt: {} tokens\n"
							 "  passkey: {}\n"
							 "  passkey token index: ~{}\n",
							 encoding.size(), passkey,
							 static_cast<int>(static_cast<float>(pos) / n_junk * encoding.size()))
			  << std::endl;

	size_t N = encoding.size();
	for (size_t pos = 0; pos < N; pos++) {
		std::cout << "\r Running passkey test..." << pos + 1 << "/" << N << std::flush;
		int token_id = encoding[pos];
		InferenceMode inferMode = pos + 1 == N ? InferenceMode::OUTPUT_LOGITS : InferenceMode::HYDRATE_KV_CACHE;
		model.forward(state, token_id, pos, inferMode);
	}
	std::cout << std::endl;
	std::cout << PROMPT_SUFFIX << std::flush;
	for (size_t pos = N; pos < N + MAX_GENERATION_STEPS; pos++) {
		int token_id = sampler.sample_argmax(state);
		std::string token_str = tokenizer.decode_one(encoding.back(), token_id);
		std::cout << token_str << std::flush;
		encoding.push_back(token_id);
		if (token_id == tokenizer.eos_id || token_id == tokenizer.eot_id) {
			break;
		}
		model.forward(state, token_id, pos);
	}
	std::cout << std::endl;
}

int run_convert(const cxxopts::ParseResult &params) {
	auto model_path = params["model"].as<std::string>();


	std::print("converting {}...\n", model_path);
	auto xalm = Xalm::load(model_path);
	auto model = Model::from_xalm(xalm, 1024);

	std::print("{}", xalm.format());
	std::print("layers: {}\n", model.config.n_layers);

	auto tt = &model.blocks[1].w1;
	Tensor tc0 = tt->convert_to(Type::F16);
	/*auto tc1 = t.convert_to(Type::F8_E3M4);
	auto tc2 = t.convert_to(Type::F8_E4M3);
	auto tc3 = t.convert_to(Type::F8_E5M2);*/

	console::print(console::bright_white, "{}", (*tt * 1073.23).format());
	console::print(console::blue, "{}", (tc0 * 1073.23).format());
	/*console::print(console::green,"{}", tc1.format());
	console::print(console::yellow,"{}", tc2.format());
	console::print(console::red,"{}", tc3.format());*/

	return 0;
}


int main(int argc, char *argv[]) {
	/*const std::map<std::string, std::function<int(const cxxopts::ParseResult&)>> commands = {
	  {"convert", run_convert},
	};
	cxxopts::Options options("Xalm", "A brief description of Xalm");
	options.add_options()
		("command", "Command to execute (run, test, convert)", cxxopts::value<std::string>())
		("m,model", "Path to the model file", cxxopts::value<std::string>())
		("i,prompt", "Prompt")
		("h,help", "Print usage");

	options.parse_positional({"command"});
	const auto result = options.parse(argc, argv);

	if (result.count("help")) {
	  std::cout << options.help() << std::endl;
	  return 0;
	}

	if (!result.count("command")) {
	  std::cout << options.help() << std::endl;
	  return 1;
	}

	std::string command = result["command"].as<std::string>();
	if (commands.find(command) == commands.end()) {
	  std::cerr << "Error: Invalid command. Allowed commands are: run, test, convert.\n";
	  return 1;
	}

	return commands.at(command)(result);

	return 0;*/


	std::string checkpoint_path = ""; // e.g. out/model.bin
	// Options
	std::string device = "cpu"; // cpu or cuda
	std::string mode = "completion"; // completion, passkey, or perplexity
	std::string prompt = "Q: What is the meaning of life? A:"; // prompt string
	std::string prompt_path; // prompt file path
	int context = 0;
	// Completion mode options
	int num_steps = 1024; // number of steps to run for
	// Passkey mode options
	int n_junk = 250; // number of junk lines to insert
	int passkey_pos = -1; // passkey position (-1 - random)

	if (argc >= 2) {
		checkpoint_path = argv[1];
	} else {
		error_usage();
	}
	for (int i = 2; i < argc;) {
		// do some basic validation
		if (i + 1 >= argc) {
			error_usage();
		} // must have arg after flag
		if (argv[i][0] != '-') {
			error_usage();
		} // must start with dash
		if (strlen(argv[i]) != 2) {
			error_usage();
		} // must be -x (one dash, one letter)

		// read in the args
		if (argv[i][1] == 'h') {
			error_usage();
		} else if (argv[i][1] == 'm') {
			if (i + 1 >= argc) {
				error_usage();
			}
			mode = argv[i + 1];
			if (std::string("completion").starts_with(mode)) {
				mode = "completion";
			} else if (std::string("passkey").starts_with(mode)) {
				mode = "passkey";
			} else if (std::string("perplexity").starts_with(mode)) {
				mode = "perplexity";
			} else {
				error_usage();
			}
			i += 2;
		} else if (argv[i][1] == 'd') {
			if (i + 1 >= argc) {
				error_usage();
			}
			device = argv[i + 1];
			if (std::string("cpu").starts_with(device)) {
				device = "cpu";
			} else if (std::string("cuda").starts_with(device)) {
				device = "cuda";
			} else {
				error_usage();
			}
			i += 2;
		} else if (argv[i][1] == 'i') {
			if (i + 1 >= argc) {
				error_usage();
			}
			prompt = argv[i + 1];
			i += 2;
		} else if (argv[i][1] == 'f') {
			if (i + 1 >= argc) {
				error_usage();
			}
			prompt_path = argv[i + 1];
			i += 2;
		} else if (argv[i][1] == 'T') {
			if (i + 1 >= argc) {
				error_usage();
			}
			context = std::stoi(argv[i + 1]);
			i += 2;
		} else if (argv[i][1] == 'l') {
			if (i + 1 >= argc) {
				error_usage();
			}
			passkey_pos = std::stoi(argv[i + 1]);
			i += 2;
		} else if (argv[i][1] == 'n') {
			if (i + 1 >= argc) {
				error_usage();
			}
			num_steps = std::stoi(argv[i + 1]);
			n_junk = num_steps;
			i += 2;
		} else {
			error_usage();
		}
	}
	int has_prompt = !prompt.empty() ? 1 : 0;
	int has_prompt_path = !prompt_path.empty() ? 1 : 0;
	if (mode == "completion" || mode == "perplexity") {
		if ((has_prompt + has_prompt_path) != 1) {
			error_usage();
		} else if (has_prompt_path) {
			std::ifstream file(prompt_path);
			if (!file.is_open()) {
				std::cerr << "Error: could not open file " << prompt_path << std::endl;
				return 1;
			}

			std::stringstream buffer;
			buffer << file.rdbuf();
			prompt = buffer.str();
		}
	} else {
		if (passkey_pos != -1 && (passkey_pos >= n_junk || passkey_pos < 0)) {
			std::cerr << "Error: passkey position must be between 0 and " << n_junk - 1 << std::endl;
			return 1;
		}
	}

	if (mode == "completion") {
		run_completion(checkpoint_path, device, prompt, context, num_steps);
		Profiler::report();
	} else if (mode == "passkey") {
		run_passkey(checkpoint_path, device, context, n_junk, passkey_pos);
	} else if (mode == "perplexity") {
		run_perplexity(checkpoint_path, device, prompt, context);
	} else if (mode == "test") {
		run_test(checkpoint_path);
		Profiler::report();
	}

	return 0;
}
