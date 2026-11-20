# Ryan Lei

I find and fix bugs where a compiler or cache **silently returns the wrong result**: a guard that misses what changed, a key that leaves out what matters, a buffer read before anything writes it. Each one below starts from a minimal repro and ends with a fail-before / pass-after regression test.

Applied Mathematics / ML at the University of Waterloo · prediction lead at **WATonomous** · research at Waterloo's **Vision and Image Processing Lab**.

## Merged upstream

### PyTorch: Dynamo dropped a user's custom `jvp` and returned wrong tangents · [pytorch#193277](https://github.com/pytorch/pytorch/pull/193277) · [`1e3e90e`](https://github.com/pytorch/pytorch/commit/1e3e90ec55)
- **Problem:** under `torch.func.jvp`, `torch.compile` inlined an `autograd.Function`'s `forward`, dropped its custom `jvp`, and returned a wrong tangent with no error, even with `fullgraph=True` ([#192985](https://github.com/pytorch/pytorch/issues/192985)).
- **Investigation:** forward-mode duals have `requires_grad=False`, so the "unsupported custom jvp" check sat behind a gate it could never pass. The bug was already in the test suite as 8 `dynamo_expected_failures` entries.
- **Fix:** moved the graph break above the `requires_grad` gate, fired it whenever a dual level is live, and added a `DUAL_LEVEL` guard so entering forward AD later recompiles.
- **Validation:** 7 new tests across `eager`, `aot_eager` and `fullgraph`; main vs. main+PR on 5 Dynamo suites shows no new failures.
- **Result:** custom `jvp` is respected; the 8 expected-failure entries were deleted because they now pass.

### PyTorch: `torch.compile` kept running a stale graph after a method was replaced · [pytorch#198124](https://github.com/pytorch/pytorch/pull/198124) · [`7004da3`](https://github.com/pytorch/pytorch/commit/7004da38d0)
- **Problem:** after `Model.m.__code__ = ...`, `Model.m = other`, or a swapped `nn.Module.forward`, compiled code kept running the old graph ([#197860](https://github.com/pytorch/pytorch/issues/197860)).
- **Investigation:** two call paths built the inlined method without a source, so no `CLOSURE_MATCH` guard was installed on its code object.
- **Fix:** build those methods from the class `__dict__` source, one guard per (class, method), deduplicated through the MRO cache. Edge cases the guard can't cover safely are documented.
- **Validation:** 10 new mutation tests in `test_recompiles`, plus tag-safe-root and `aot_compile` save/load tests; 1109 passed across 8 suites after rebase.
- **Result:** merged. Measured guard cost: +4% guard-eval time with one shared block class, +39% with 20 distinct classes.

### Coder: an https login token was sent in plaintext to the http URL · [coder#29764](https://github.com/coder/coder/pull/29764)
- **Problem:** the CLI keyed keyring credentials by host only, so `http://` and `https://` on the same host shared one token, and an http logout signed out the https session ([#29290](https://github.com/coder/coder/issues/29290)).
- **Investigation:** reproduced against Windows Credential Manager and captured the stored blob and cross-scheme reads before the fix.
- **Fix:** key entries by origin. https keeps the bare-host key older CLIs already write, so existing sessions keep working; http gets its own key; other schemes are rejected.
- **Validation:** unit tests for normalization and cross-scheme read/delete; real before/after transcripts against Credential Manager; `go vet` on Windows, Linux and macOS.
- **Result:** a token for one scheme is no longer readable or deletable through the other; http and https logins on the same host can coexist.

### TileLang: a general check for buffers read before they are written · [tilelang#2956](https://github.com/tile-ai/tilelang/pull/2956) · +1022 lines, 28 tests
- **Problem:** a `T.gemm` accumulator that nothing initializes gives correct output on one machine and NaN on another ([#2936](https://github.com/tile-ai/tilelang/issues/2936)).
- **Investigation:** my first PR ([#2939](https://github.com/tile-ai/tilelang/pull/2939)) was gemm-specific, and the maintainer asked for the general version. Tile ops expose *may-read* regions, but this check needs *must-read*, and the common `clear_accum=(k == 0)` idiom is exactly where the two differ.
- **Fix:** `VerifyBufferInit`, a warning-only pass placed before `LowerTileOp` so tile-op writes are still visible. It adds one virtual (`GetReadBeforeWriteRegions`) that only the gemm ops override, and uses scope-aware ordering so warp-specialized producer/consumer kernels stay quiet.
- **Validation:** 997 language tests with zero warnings; all 285 example files scanned.
- **Result:** shipped **on by default**. It found 3 real read-before-writes in 2 examples, present since Jan and May 2025; on the first one, seeding the uninitialized buffer with `+inf` turns all 524,288 outputs into NaN.
- **Follow-up (ownership):** I audited my own pass with hand-written kernels, filed [#3619](https://github.com/tile-ai/tilelang/issues/3619) (no source location for reads outside tile ops), and fixed it in [#3620](https://github.com/tile-ai/tilelang/pull/3620) (in review).

### In review
- **NVIDIA Warp:** CUDA-graph (APIC) capture replays the wrong overload when one capture launches several overloads of a generic kernel. I filed issue [#2140](https://github.com/NVIDIA/warp/issues/2140) and opened fixes [#2141](https://github.com/NVIDIA/warp/pull/2141) and [#2142](https://github.com/NVIDIA/warp/pull/2142).
- **Cua:** Wayland presentation-timestamp evidence for the computer-use driver: [cua#4236](https://github.com/trycua/cua/pull/4236), for [#4012](https://github.com/trycua/cua/issues/4012).
  - **Mechanism:** a fixture surface requests `wp_presentation.feedback` before each commit and accepts feedback only for its own update ID, so late or unknown callbacks can't claim the wrong row.
  - **Clock domains:** it never subtracts timestamps across clock domains. A foreign clock is reported as a typed `clock_mismatch`, not as a latency.
  - **Validation:** 25 fixture tests, including wrong and late feedback correlation, plus a native headless Sway 1.9 cell with 2 verified presentations, 1 no-op and 1 discarded superseded update.
- **WATonomous:** MTR trajectory-prediction inference, lifecycle and quality harness: [wato_monorepo#476](https://github.com/WATonomous/wato_monorepo/pull/476).

## Projects

### [comma Controls Challenge](https://github.com/RyanL2/commacontrol): tied #1 at 6.880, then showed why the top of the board isn't measuring control
- **Problem:** minimize lateral-control cost on comma's learned-physics simulator.
- **Investigation:** I built a GPU/CUDA-graph optimization pipeline behind a strict parity gate, then found that inside the scored window the cost is a convex quadratic that doesn't depend on the neural model at all.
- **Implementation:** a closed-form per-segment optimum, checked against an exact min-plus DP over all 1024 output bins (delta 0 across 5000 segments).
- **Validation:** pre-registered ablation thresholds, including an 18.99% improvement I rejected because it broke a per-segment cap I had set in advance. Negative results are kept.
- **Result:** 6.880 (tied #1), a 6.18 hard floor for the metric, and an honest no-injection controller at 39.9 against PID's ~68. [Write-up](https://ryleitianebao.medium.com/i-tied-for-1-on-a-self-driving-benchmark-by-building-something-that-didnt-need-to-exist-07400657fef7).

### [Dry Run](https://github.com/RyanL2/dry-run): judge a coding agent's shell command by what it actually did, not by its text
- **Problem:** command-text allowlists can't tell `rm -rf build/` from a script that deletes your home directory.
- **Implementation:** a Claude Code `PreToolUse` hook plus a daemon. It runs each command in a copy-on-write, network-less shadow of the workspace, extracts the effects, applies harm-policy rules, then commits exactly the reviewed change.
- **Validation:** 16 isolation canaries, and a benchmark with template-disjoint splits plus pre-registered experiment cards (≥3 paired seeds per verdict).
- **Status:** core v0.1 works on Linux/WSL2; the learned judge is still in research and not yet promoted into the shipped decision cascade.

## Tools

C++ · Python · PyTorch (Dynamo) · CUDA · TVM/TileLang · Go · ROS 2 · TensorRT · ONNX · Linux

[LinkedIn](https://linkedin.com/in/lei-ryan) · [Email](mailto:ryleitianebao@gmail.com)
