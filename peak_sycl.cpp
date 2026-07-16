#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace {

void usage(const char *program) {
  std::cout << "Usage:\n"
            << "  " << program
            << " mem [--size-mib N] [--iterations N]\n"
            << "  " << program
            << " compute [--work-items N] [--rounds N] [--iterations N]\n"
            << "  " << program
            << " gemv [--size N | --m M --k K] [--type bf16|fp16]\n"
            << "  " << program
            << " gemm [--size N | --m M --n N --k K] [--type bf16|fp16]\n";
}

[[noreturn]] void dispatch(const char *program, char **argv) {
  const auto executable = std::filesystem::canonical("/proc/self/exe");
  const auto target = executable.parent_path() / program;
  argv[0] = const_cast<char *>(target.c_str());
  execv(target.c_str(), argv);
  throw std::runtime_error("cannot execute " + target.string());
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 2) {
      usage(argv[0]);
      throw std::invalid_argument("missing benchmark mode");
    }

    const std::string mode = argv[1];
    if (mode == "-h" || mode == "--help") {
      usage(argv[0]);
      return 0;
    }

    // Remove the mode from argv so each benchmark retains its existing option
    // parser, diagnostics, and defaults.
    if (mode == "mem") dispatch("peak_mem_2d", argv + 1);
    if (mode == "compute") dispatch("peak_compute_dpas", argv + 1);
    if (mode == "gemv") dispatch("gemv", argv + 1);
    if (mode == "gemm") dispatch("gemm", argv + 1);
    throw std::invalid_argument("unknown benchmark mode: " + mode);
  } catch (const std::exception &error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }
}
