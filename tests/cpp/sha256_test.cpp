#include "microsimulator/sha256.hpp"

#include <cassert>
#include <string>

int main() {
  using microsimulator::digests_equal;
  using microsimulator::sha256_hex;

  // FIPS 180-4 examples, plus lengths on each side of the padding boundaries.
  assert(sha256_hex("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  assert(sha256_hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  assert(sha256_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") ==
         "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
  assert(sha256_hex(std::string(55, 'a')) ==
         "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318");
  assert(sha256_hex(std::string(56, 'a')) ==
         "b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a");
  assert(sha256_hex(std::string(64, 'a')) ==
         "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb");
  assert(sha256_hex(std::string(1'000'000, 'a')) ==
         "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");

  assert(digests_equal("abc", "abc"));
  assert(!digests_equal("abc", "abd"));
  assert(!digests_equal("abc", "abcd"));
  assert(digests_equal("", ""));
  return 0;
}
