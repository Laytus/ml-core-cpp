# Build Notes – ML Core

## Purpose

This document records the build conventions for **ML Core**.

Its goal is to keep the project build process:
- clear
- repeatable
- easy to debug
- stable as the repository grows

This is not a full tutorial on CMake or Eigen.

It is the project-specific build reference.

---

## Build System

The project uses:
- **CMake**
- **C++17**
- **Eigen** as the matrix library

Current executable targets:
- `ml_core_app`
- `ml_core_tests`

---

## Current Build Targets

## `ml_core_app`
Purpose:
- clean app entrypoint
- stable project entry executable
- should remain minimal

Current source:
- `app/main.cpp`

## `ml_core_tests`
Purpose:
- structured manual validation runner
- phase sanity checks
- temporary but organized testing/demo executable

Current source:
- `app/test_runner.cpp`

---

## Compiler Standard

The project currently targets:

```cmake
CMAKE_CXX_STANDARD 17
```

And should keep:
- `CMAKE_CXX_STANDARD_REQUIRED ON`
- `CMAKE_CXX_EXTENSIONS OFF`

This keeps the project predictable and avoids unnecessary compiler-specific extensions.

---

## Eigen Integration

The project uses **Eigen** for matrix and vector operations.

### Why Eigen is used
Eigen is included because:
- matrix operations are central to later ML phases
- the project goal is ML, not building a matrix library from scratch
- using Eigen avoids wasting time on infrastructure that is not the main learning target

### Include style
Use:

```cpp
#include <Eigen/Dense>
```

when dense matrix/vector operations are needed.

---

## Installing Eigen on macOS

With Homebrew:

```bash
brew install eigen
```

On Apple Silicon, the headers are commonly available under:

```text
/opt/homebrew/include/eigen3
```

On Intel Macs, it is often:

```text
/usr/local/include/eigen3
```

---

## CMake Eigen Discovery

The project uses:

```cmake
find_package(Eigen3 REQUIRED NO_MODULE)
```

and links with:

```cmake
target_link_libraries(<target> PRIVATE Eigen3::Eigen)
```

This is the preferred project-level integration approach.

---

## Typical Build Commands

From the project root:

```bash
cmake -S . -B build
cmake --build build
```

Run executables with:

```bash
./build/ml_core_app
./build/ml_core_tests
```

---

## Clean Reconfigure

When CMake configuration changes significantly, do a clean reconfigure:

```bash
rm -rf build
cmake -S . -B build
cmake --build build
```

This is especially useful after:
- target changes
- include path changes
- Eigen/CMake configuration changes

---

## VS Code Notes

A project can build successfully with CMake while VS Code still shows include errors.

This usually means IntelliSense is out of sync, not that the real build is broken.

### Recommended VS Code setup
Use:
- **CMake Tools**
- **C/C++** extension

And set:

```json
{
  "C_Cpp.default.configurationProvider": "ms-vscode.cmake-tools"
}
```

This lets VS Code derive include paths from CMake.

### If IntelliSense still fails
Useful fixes:
- `CMake: Delete Cache and Reconfigure`
- `C/C++: Reset IntelliSense Database`

### Important note
If VS Code shows a wrong path like `/Users/user/...` while your real path is `/Users/heber/...`, that likely indicates stale workspace metadata or stale IntelliSense configuration.

---

## Build Philosophy for This Project

The build setup should stay:
- simple
- explicit
- scalable

### Rule 1
Do not overcomplicate CMake early.

Add complexity only when the project structure truly requires it.

### Rule 2
Keep reusable library code separate from executable wiring.

### Rule 3
The build should reflect the repository structure:
- reusable code in `include/` + `src/`
- executable entrypoints in `app/`

### Rule 4
Prefer incremental clean expansion over clever but fragile CMake abstractions.

---

## Expected Build Evolution

At the current stage, two executables are enough:

- `ml_core_app`
- `ml_core_tests`

Later, the build may expand to include:
- more source files from `src/`
- reusable library targets if useful
- additional experiment executables only if clearly justified

### Important rule
Do not create many targets too early without a strong reason.

Keep the build understandable.

---

## Build Hygiene Rules

- keep `main.cpp` minimal
- keep validation code in `test_runner.cpp`
- do not put reusable logic only inside executable source files
- when reusable code appears, move it into `include/` + `src/`
- avoid hidden include-path assumptions
- keep target definitions explicit

---

## Current Phase 0 Build Success Criteria

Phase 0 build setup is considered successful when:
- Eigen is found cleanly by CMake
- both executables build successfully
- `ml_core_app` runs
- `ml_core_tests` runs
- VS Code/editor issues, if any, are understood as editor config issues rather than actual build failures

---

## Final Principle

The build system should help the project move faster, not become a project of its own.

For ML Core, the build exists to support:
- serious ML implementation
- clean project structure
- reliable iteration across phases

That is its role.
