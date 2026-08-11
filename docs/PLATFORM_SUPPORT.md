# Platform Support Policy

## 1. Release target

The primary release environment is **native Windows 11 x86-64 with CPython 3.11**. A release cannot claim repository-wide support until its core installation, CLI, repository-owned tests, package build, and declared release-supported workflows pass in a clean primary environment.

Linux x86-64 with CPython 3.11 is a secondary compatibility environment. Linux CI remains required because some authoritative upstream packages and historical parity protocols are Linux-specific, but Linux success cannot substitute for the Windows release gate.

macOS is currently unsupported. This is a statement about tested release support, not a claim that every command necessarily fails on macOS.

## 2. CPU and GPU profiles

The minimal library, metadata commands, dataset tooling, table validation, and CPU-capable diagnostics MUST work without a GPU on the primary Windows environment.

GPU-dependent runs use separate hardware profiles. A result produced under one GPU profile MUST NOT be generalized to another profile without compatibility evidence. The first P4 Windows GPU pilot targets exactly:

- native Windows 11 x86-64;
- CPython 3.11;
- NVIDIA GeForce RTX 5080 with 16 GiB-class VRAM;
- a recorded NVIDIA driver and CUDA runtime;
- a checksum-locked TabPFN checkpoint pair; and
- a fully frozen Python dependency set.

This pilot does not make an RTX 5080 a mandatory dependency of the repository. It qualifies one hardware-specific Global Utility execution profile.

## 3. Evidence interpretation

Historical Linux/Python 3.11 evidence remains immutable and valid for the claim it originally established. It can support source or adapter parity when the declared upstream runtime is Linux-specific. It does not establish current Windows release support.

The following claims are independent:

- `native-parity-validated`: parity against the declared upstream target in the recorded environment;
- `benchmark-eligible`: admission under the relevant scientific protocol; and
- `release-supported`: installation, compatibility, documentation, and maintenance support on the primary Windows environment.

Changing the primary release platform does not erase historical evidence or automatically promote or demote an adapter's parity status. It creates a new Windows release-compatibility gate.

## 4. CI and release gates

Every release candidate MUST pass:

1. Windows 11/Python 3.11 core CI, packaging, and clean-install smoke tests;
2. Windows-native validation for every component claimed as release-supported;
3. the declared Windows GPU profile for every GPU-dependent official result;
4. Linux/Python 3.11 secondary compatibility CI; and
5. explicit documentation of components that retain Linux-only upstream parity evidence but lack Windows release support.

Historical evidence files MUST NOT be edited to reflect this policy change. New Windows evidence receives a new protocol or environment identity.

