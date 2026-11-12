# Third-party sources

Single-header libraries compiled into the core. Only `.cpp` files under `cpp/core` include them, so the public headers in `cpp/include` and programs that link the core need neither these files nor any extra library.

| Library | Version | File | SHA-256 | License |
|---|---|---|---|---|
| [JSON for Modern C++](https://github.com/nlohmann/json) | 3.12.0 | `nlohmann/json.hpp` | `aaf127c04cb31c406e5b04a63f1ae89369fccde6d8fa7cdda1ed4f32dfc5de63` | MIT, `nlohmann/LICENSE.MIT` |
| [PicoSHA2](https://github.com/okdshin/PicoSHA2) | 1.0.1 | `picosha2/picosha2.h` | `b13c180161ffac8d0adc81e033e493c409457c4d1258ab9781ac80579ba3bdd8` | MIT, `picosha2/LICENSE` |

The files are byte-identical to the upstream releases. To update one, replace the file with the new release and record its version and checksum here.
