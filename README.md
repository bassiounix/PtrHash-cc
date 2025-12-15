# PtrHash-cc

This is a port (rewrite) of `PtrHash` (<https://github.com/RagnarGrootKoerkamp/PtrHash>) in C++.
(commit id [0c971867c903504892527a1747703e3f75eaac2c](https://github.com/RagnarGrootKoerkamp/PtrHash/tree/0c971867c903504892527a1747703e3f75eaac2c)).
The port contains a parallel implementation of `FxHash`.
This repository is provided as is with no guarantees.
The main purpose for this port is to use it in LLVM libc's `wctype` proposal to make a compile-time hash table. LLVM codebase is not written in Rust.
Related RFC: <https://discourse.llvm.org/t/rfc-libc-wctype-header-implementation/88941>

## Future Considerations

- We need to check alternatives to standard C's `rand` implementation.
- Integrate the code into LLVM libc's code with the necessary modifications.
- Do I need this for other projects in the future? We will see 😄.

## License

This code is under the GPLv2 license.
All rights reserved to the author of the repository.
