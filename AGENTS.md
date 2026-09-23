# Agent instructions

- Write commit descriptions as easy-to-read, one-line capitalized imperative sentences.
- Follow the project's `.clang-format` configuration for code style. Prefer `snake_case` for function names, variable names, and source code file names. Use `.cpp` and `.hpp` for C++ files, and `.cu` and `.cuh` for CUDA files.
- Prefer a blank line before and after `if`, `for`, and `while` constructs to separate them from surrounding statements.
- Do not use emoji anywhere in the project.
- Modify `README.md` only to resolve inconsistencies; do not update it automatically when adding a feature.
- Use [fsossai/timers](https://github.com/fsossai/timers) for timing. Declare named global `Stopwatch` variables with an `sw_` prefix, and let the library print their results automatically. Use `ScopedTimer` only when a timed section has multiple exit points; when timing a whole function, create one `ScopedTimer` at its entry. For other sections, call the stopwatch's `start()` and `stop()` methods directly. Name nested timers with dotted paths, such as `phase1.subphase2`. Label non-obvious `Stopwatch` constructor arguments with `/*arg=*/` comments, but do not label the name argument.
