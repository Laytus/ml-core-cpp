# Build Notes – ML Core

## Purpose

This document records the final build conventions for **ML Core**.

Its goal is to keep the project build process:

- clear
- repeatable
- easy to debug
- compatible with automated testing
- aligned with the final repository structure

This is not a general tutorial on CMake, Eigen, Catch2, or CTest.

It is the project-specific build reference.

---

## Build System

The project uses:

- **CMake**
- **C++17**
- **Eigen** as the matrix library
- **Catch2** for automated C++ tests
- **CTest** for test discovery and execution
- **GitHub Actions** for continuous integration

The final executable/test targets are:

- `ml_core_app`
- `ml_core_validation`
- `ml_core_unit_tests` when `BUILD_TESTING=ON`

---

## Build Targets

### `ml_core_app`

Purpose:

- clean application entrypoint
- stable project entry executable
- intentionally minimal

Current source:

```text
app/main.cpp
```

### `ml_core_validation`

Purpose:

- structured manual validation runner
- phase sanity checks
- behavior studies
- practical workflow validation

Current source:

```text
app/test_runner.cpp
```

The target also links the phase-specific validation code under `experiments/`.

This target is intentionally separate from the automated test suite.

### `ml_core_unit_tests`

Purpose:

- automated deterministic correctness checks
- model and utility validation
- API and input-validation checks
- CI-compatible regression protection

Automated test sources live under:

```text
tests/
```

The test executable is built when:

```text
BUILD_TESTING=ON
```

Catch2 test cases are registered with CTest through `catch_discover_tests`.

---

## Compiler Standard

The project targets:

```cmake
CMAKE_CXX_STANDARD 17
```

and keeps:

```cmake
CMAKE_CXX_STANDARD_REQUIRED ON
CMAKE_CXX_EXTENSIONS OFF
```

This keeps the project predictable and avoids unnecessary compiler-specific extensions.

---

## Eigen Integration

The project uses **Eigen** for matrix and vector operations.

### Why Eigen is used

Eigen is included because:

- matrix operations are central to the ML implementations
- the project goal is ML, not rebuilding a matrix library
- using Eigen keeps the implementation focused on algorithms and workflows

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

On Intel Macs, they are commonly available under:

```text
/usr/local/include/eigen3
```

---

## CMake Eigen Discovery

The project uses:

```cmake
find_package(Eigen3 REQUIRED NO_MODULE)
```

and links through:

```cmake
Eigen3::Eigen
```

The reusable `ml_core` library exposes Eigen as a public dependency.

---

## Catch2 and CTest Integration

CTest is enabled from the root `CMakeLists.txt`:

```cmake
include(CTest)
```

When `BUILD_TESTING=ON`, Catch2 is fetched at the pinned version used by the project and the `tests/` directory is added to the build.

The current Catch2 integration uses:

```text
v3.16.0
```

The automated test executable is:

```text
ml_core_unit_tests
```

and individual Catch2 test cases are discovered through CTest.

### Test design

The automated suite should remain:

- fast
- deterministic
- self-contained
- independent from large real datasets
- focused on correctness, validation, and regression protection

Behavior studies and practical workflows remain outside the automated suite.

---

## Typical Build Commands

From the project root:

```bash
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build
```

Run the automated test suite with:

```bash
ctest --test-dir build --output-on-failure --no-tests=error
```

List discovered tests with:

```bash
ctest --test-dir build -N
```

Run the minimal application entrypoint with:

```bash
./build/ml_core_app
```

Run the manual validation layer with:

```bash
./build/ml_core_validation
```

---

## Clean Reconfigure

When CMake configuration changes significantly, use a clean reconfigure:

```bash
rm -rf build
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build
```

Then validate:

```bash
ctest --test-dir build -N
ctest --test-dir build --output-on-failure --no-tests=error
./build/ml_core_validation
```

A clean reconfigure is especially useful after:

- target changes
- dependency changes
- test-integration changes
- include-path changes
- CMake configuration changes

---

## Automated Tests vs Manual Validation vs Experiments

The final project intentionally separates three validation responsibilities.

### Automated tests

Location:

```text
tests/
```

Execution:

```bash
ctest --test-dir build --output-on-failure --no-tests=error
```

Use for:

- deterministic correctness checks
- known mathematical behavior
- option/input validation
- API misuse
- model invariants
- regression protection

### Manual validation

Entry executable:

```text
ml_core_validation
```

Use for:

- selected sanity workflows
- phase-level manual checks
- convenient development-time validation

### Experiments

Location:

```text
experiments/
```

Use for:

- model behavior studies
- optimizer comparisons
- hyperparameter sweeps
- real-dataset workflows
- generated experiment outputs

These layers should remain separate.

---

## Continuous Integration

The repository uses GitHub Actions.

Workflow:

```text
.github/workflows/ci.yml
```

The current CI job:

1. checks out the repository
2. installs Ninja and Eigen
3. configures CMake with `BUILD_TESTING=ON`
4. builds the project
5. runs CTest

The CI test command is:

```bash
ctest --test-dir build --output-on-failure --no-tests=error
```

The `--no-tests=error` option is intentional: CI must fail if a configuration regression causes CTest to discover zero tests.

The default CI path runs the automated suite only.

`ml_core_validation` and heavier experiment workflows remain outside the default CI job.

---

## VS Code Notes

A project can build successfully with CMake while VS Code still shows include errors.

This usually means IntelliSense is out of sync, not that the real build is broken.

### Recommended VS Code setup

Use:

- **CMake Tools**
- **C/C++** extension

and configure:

```json
{
  "C_Cpp.default.configurationProvider": "ms-vscode.cmake-tools"
}
```

This lets VS Code derive include paths from CMake.

### If IntelliSense still fails

Useful fixes include:

- `CMake: Delete Cache and Reconfigure`
- `C/C++: Reset IntelliSense Database`

---

## Build Philosophy

The build setup should stay:

- simple
- explicit
- reproducible
- aligned with the repository structure

### Rule 1

Keep reusable library code separate from executable wiring.

### Rule 2

Keep automated correctness tests separate from manual validation and experiments.

### Rule 3

Prefer explicit CMake target definitions over unnecessarily clever abstractions.

### Rule 4

CI should reproduce the same essential configure, build, and automated-test workflow used locally.

---

## Final Build Structure

The final build separates three responsibilities:

```text
ml_core_app
    minimal application entrypoint

ml_core_validation
    manual sanity checks, behavior studies, and practical validation

ml_core_unit_tests
    automated Catch2 tests executed through CTest
```

This separation should remain stable.

---

## Build Hygiene Rules

- keep `main.cpp` minimal
- keep manual validation wiring in `test_runner.cpp`
- keep automated correctness tests under `tests/`
- keep behavior studies and practical workflows under `experiments/`
- do not put reusable logic only inside executable/test sources
- move reusable logic into `include/` + `src/`
- avoid hidden include-path assumptions
- keep target definitions explicit
- keep automated tests deterministic and fast
- do not add heavy practical workflows to CI without a clear reason

---

## Final Build Success Criteria

The build setup is considered healthy when:

- Eigen is found cleanly by CMake
- `ml_core_app` builds
- `ml_core_validation` builds and runs
- `ml_core_unit_tests` builds when `BUILD_TESTING=ON`
- CTest discovers the expected automated test suite
- all automated tests pass
- CI fails if zero tests are discovered
- CI reproduces the configure, build, and automated-test workflow successfully

For the current repository state, CTest should discover the established automated test suite maintained under `tests/`.

---

## Final Principle

The build system should support the project without becoming a project of its own.

For ML Core, the build exists to support:

- serious ML implementation
- clear separation of responsibilities
- deterministic automated testing
- manual validation and experimentation
- reproducible CI

That is its role.
