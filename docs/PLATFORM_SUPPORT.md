# Platform Support Policy

## 1. Release family and exact target

The primary release family is **Windows x86-64 with CPython 3.11**. The exact consumer release target is **native Windows 11 x86-64 with CPython 3.11**.

These are related but distinct claims. GitHub's `windows-latest` hosted runner currently supplies a Windows Server image. Its success establishes compatibility with the primary Windows family, but MUST NOT be described as exact Windows 11 qualification. Exact qualification requires a recorded native or self-hosted Windows 11 run that identifies the OS edition and build, architecture, Python version, repository commit, and installed dependency set.

The repository's portable platform classifier intentionally recognizes the Windows/Python family only. Python's standard platform APIs cannot reliably distinguish every Windows 11 consumer installation from every Windows Server installation that shares its kernel build. The classifier therefore MUST NOT be used as proof of the exact OS edition.

Linux x86-64 with CPython 3.11 is a required secondary compatibility environment. Some authoritative upstream packages and historical parity protocols are Linux-specific, but Linux success cannot substitute for either Windows-family CI or the exact Windows 11 release gate.

macOS is currently unsupported. This describes tested release support; it does not assert that every command necessarily fails on macOS.

## 2. CPU and GPU profiles

The minimal library, metadata commands, dataset tooling, table validation, and CPU-capable diagnostics MUST work without a GPU in the primary Windows family and on the exact Windows 11 target before release.

GPU-dependent runs use separate hardware profiles. A result produced under one GPU profile MUST NOT be generalized to another profile without compatibility evidence. The first P4 Windows GPU pilot targets exactly:

- native Windows 11 x86-64;
- CPython 3.11;
- NVIDIA GeForce RTX 5080 with 16 GiB-class VRAM;
- a recorded NVIDIA driver and CUDA runtime;
- a checksum-locked TabPFN checkpoint pair; and
- a fully frozen Python dependency set.

This pilot does not make an RTX 5080 a mandatory dependency of the repository. It qualifies one hardware-specific Global Utility execution profile.

## 3. Evidence interpretation

Historical Linux/Python 3.11 evidence remains immutable and valid for the claim it originally established. It can support source or adapter parity when the declared upstream runtime is Linux-specific. It does not establish Windows release support.

The following evidence scopes MUST remain separate:

- **Hosted Windows-family CI:** portable installation, test, CLI, and packaging compatibility on the recorded hosted Windows image;
- **Exact Windows 11 qualification:** release behavior on the declared native Windows 11 target; and
- **Hardware-profile qualification:** behavior under one fully identified CPU/GPU and dependency profile.

The following lifecycle claims are also independent:

- `native-parity-validated`: parity against the declared upstream target in the recorded environment;
- `benchmark-eligible`: admission under the relevant scientific protocol; and
- `release-supported`: installation, compatibility, documentation, and maintenance support on the exact release target.

Changing the primary release platform does not erase historical evidence or automatically promote or demote an adapter's parity status. It creates new Windows-family and exact-target compatibility gates.

## 4. CI and release gates

Every release candidate MUST pass:

1. hosted Windows/Python 3.11 primary-family core CI, packaging, and clean-install smoke tests;
2. a recorded native or self-hosted Windows 11/Python 3.11 exact-target run for the same candidate commit;
3. exact-target validation for every component claimed as `release-supported`;
4. the declared Windows GPU profile for every GPU-dependent official result;
5. Linux/Python 3.11 secondary compatibility CI; and
6. explicit documentation of components that retain Linux-only upstream parity evidence but lack Windows release support.

Historical evidence files MUST NOT be edited to reflect this policy change. New evidence receives a new protocol or environment identity. Hosted Windows Server evidence and exact Windows 11 evidence MUST be named and reported separately.
