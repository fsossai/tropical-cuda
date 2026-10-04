# Agent instructions

- Write commit descriptions as easy-to-read, one-line capitalized imperative sentences.
- Do not add AI agents as co-authors of commits
- Follow the project's `.clang-format` configuration for code style. Prefer `snake_case` for function names, variable names, and source code file names. Use `.cpp` and `.hpp` for C++ files, and `.cu` and `.cuh` for CUDA files.
- Use unqualified `size_t`, `uintN_t`, and `intN_t` type names; do not prefix them with `std::`.
- Prefer a blank line before and after `if`, `for`, and `while` constructs to separate them from surrounding statements.
- Add a comment of at most 2 lines describing the purpose of each nontrivial class and function.
- Do not use emoji anywhere in the project.
- Modify `README.md` only to resolve inconsistencies; do not update it automatically when adding a feature.
