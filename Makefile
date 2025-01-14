# Compiler and tools
CC := clang
CXX := clang++
MAKEFLAGS += -r -j 8

UNAME := $(shell uname)

BUILD := build
SRC_DIR := src
VENDOR_DIR := vendor
ASM_DIR := $(BUILD)/asm
BIN_DIR := .

CFLAGS := -g -Wall -Wpointer-arith -march=native -O3 -Werror -I$(VENDOR_DIR) -std=c++23 -fcolor-diagnostics
LDFLAGS := -lm

ifeq ($(UNAME), Darwin)	# MAC OS
    CFLAGS += -mcpu=native -Xpreprocessor -fopenmp -I /opt/homebrew/opt/libomp/include
    LDFLAGS += -lomp -L /opt/homebrew/opt/libomp/lib
else # LINUX
    LDFLAGS += -fopenmp
    CFLAGS += -mf16c -mavx2 -mfma
endif

# compile .c, .cpp, .cu files
SOURCES=$(filter-out src/test.cpp,$(wildcard src/*.c))
SOURCES+=$(filter-out src/test.cpp,$(wildcard src/*.cc))
SOURCES+=$(filter-out src/test.cpp,$(wildcard src/*.cpp))
#SOURCES+=$(filter-out src/test.cpp,$(wildcard src/*.cu))
SOURCES+=$(wildcard vendor/*.c)
SOURCES+=$(wildcard vendor/*.cc)
SOURCES+=$(wildcard vendor/*.cpp)
#SOURCES+=$(wildcard vendor/*.cu)

# Define test sources separately
TEST_SOURCES=$(filter-out src/main.cpp,$(wildcard src/*.c))
TEST_SOURCES+=$(filter-out src/main.cpp,$(wildcard src/*.cc))
TEST_SOURCES+=$(filter-out src/main.cpp,$(wildcard src/*.cpp))
TEST_SOURCES+=$(wildcard vendor/*.c)
TEST_SOURCES+=$(wildcard vendor/*.cc)
TEST_SOURCES+=$(wildcard vendor/*.cpp)

OBJECTS=$(SOURCES:%=$(BUILD)/%.o)
TEST_OBJECTS=$(TEST_SOURCES:%=$(BUILD)/%.o)
ASM_FILES=$(patsubst %.cpp,$(ASM_DIR)/%.s,$(filter %.cpp,$(SOURCES)))
TEST_ASM_FILES=$(patsubst %.cpp,$(ASM_DIR)/%.s,$(filter %.cpp,$(TEST_SOURCES)))

BINARY=$(BUILD)/main
TEST_BINARY=$(BUILD)/test

#LDFLAGS+=-lcudart
#
#ifneq (,$(wildcard /usr/local/cuda))
#  LDFLAGS+=-L/usr/local/cuda/lib64
#endif
#
#CUFLAGS+=-O2 -lineinfo -Ivendor
#CUFLAGS+=-allow-unsupported-compiler # for recent CUDA versions
#
#ifeq ($(CUARCH),)
#  CUFLAGS+=-gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90 --threads 2
#else
#  CUFLAGS+=-arch=$(CUARCH)
#endif

all: $(BINARY) asm

test: $(TEST_BINARY) test-asm

# Target to build just assembly files
asm: $(ASM_FILES)

test-asm: $(TEST_ASM_FILES)

format:
	clang-format -i src/*

$(BINARY): $(OBJECTS)
	$(CXX) $^ $(LDFLAGS) -o $@

$(TEST_BINARY): $(TEST_OBJECTS)
	$(CXX) $^ $(LDFLAGS) -o $@

# Rule to generate assembly for cpp files
$(ASM_DIR)/%.s: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $< $(CFLAGS) -S -o $@

$(BUILD)/%.c.o: %.c
	@mkdir -p $(dir $@)
	$(CXX) $< $(CFLAGS) -c -MMD -MP -o $@

$(BUILD)/%.cpp.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $< $(CFLAGS) -c -MMD -MP -o $@

$(BUILD)/%.cc.o: %.cc
	@mkdir -p $(dir $@)
	$(CXX) $< $(CFLAGS) -c -MMD -MP -o $@

$(BUILD)/%.cu.o: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) $< $(CUFLAGS) -c -MMD -MP -o $@

-include $(OBJECTS:.o=.d)
-include $(TEST_OBJECTS:.o=.d)

clean:
	rm -rf $(BUILD)

.PHONY: all clean format test asm test-asm