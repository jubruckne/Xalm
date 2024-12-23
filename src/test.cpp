#include "fmt/format.h"

#include "tensor.h"

int main(int argc, char* argv[]) {
  Tensor t = Tensor("a", Type::F32, {1024, 1024});
  return 0;
}