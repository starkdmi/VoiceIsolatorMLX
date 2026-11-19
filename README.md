# VoiceIsolatorMLX

The [Inworld Voice Isolator](https://huggingface.co/inworld/voice-isolator)
(48 kHz speech isolation, Apache-2.0) on Apple silicon with
[MLX Swift](https://github.com/ml-explore/mlx-swift): the offline and the
realtime model, the streaming session, STFT, resampling, checkpoint conversion
and audio I/O, all in Swift and Metal.

- **Offline** (`voice-isolator-v1`): every frame sees its whole 8 s window;
  long inputs render in windows crossfaded over 2 s.
- **Realtime** (`voice-isolator-v1-realtime`): causal, 0/20/40 ms look-ahead,
  pushed in pieces of any size, with KV-cached attention, carried Mamba state
  and the reference's windowed two-lane crossfade.

Both match the PyTorch reference within 1.1e-6 (max absolute error of the
output waveform, float32).

## Installation

```swift
.package(url: "https://github.com/starkdmi/VoiceIsolatorMLX", from: "0.1.0")
```

with the product `VoiceIsolatorMLX`. Requires macOS 14 / iOS 17 on Apple
silicon and mlx-swift 0.29.1 through 0.32.x. MLX compiles Metal shaders, so
the package builds with Xcode (`xcodebuild`), not SwiftPM's command line; Xcode
26 and later need the Metal Toolchain component once:

```bash
xcodebuild -downloadComponent MetalToolchain
```

## Models

| repository | model | files |
| --- | --- | --- |
| [starkdmi/VoiceIsolator_v1_48K_MLX](https://huggingface.co/starkdmi/VoiceIsolator_v1_48K_MLX) | offline | `model_fp16.safetensors` (152 MiB), `model_fp32.safetensors` (272 MiB) |
| [starkdmi/VoiceIsolator_v1_Realtime_48K_MLX](https://huggingface.co/starkdmi/VoiceIsolator_v1_Realtime_48K_MLX) | realtime | `model_fp16.safetensors` (139 MiB), `model_fp32.safetensors` (253 MiB) |

`Network(contentsOf:)` and `VoiceIsolator(contentsOf:)` also load the released
PyTorch checkpoints (`denoiser.safetensors`, `denoiser_realtime.safetensors`)
and convert them in memory; `Weights.convert` writes converted files
(float32, float16, bfloat16 or 8-bit).

## Usage

```swift
import VoiceIsolatorMLX

// offline: any length, any sample rate; float16 compute by default
let isolator = try VoiceIsolator(contentsOf: offlineURL)
let audio = try AudioIO.read(inputURL)
let voice = try isolator.isolate(audio.mono, sampleRate: audio.sampleRate) { done in
    print("\(Int(done * 100))%")
}
let background = zip(audio.mono, voice).map { $0 - $1 }

// streaming: 48 kHz mono in pieces of any size
let live = try VoiceIsolator(contentsOf: realtimeURL, mode: .streaming, lookaheadFrames: 2)
let session = try live.stream()
let ready = try session.push(piece)   // samples that became final
let rest = try session.flush()
```

- `isolate` resamples to 48 kHz and back (torchaudio's `kaiser_best` sinc
  filter), reports progress after every window and throws
  `CancellationError` between windows when its task is cancelled.
- `dtype:` selects the compute precision (`.float16` default, `.float32`,
  `.bfloat16`; `nil` keeps the stored one). The STFT, the scan state, the Mamba
  output projections, softmax and norm statistics stay float32.
- `Network`, `VoiceIsolator` and `IsolationStream` are not reentrant: confine
  each to one actor or queue. Separate instances may run concurrently.
- `Network.maxTokens` (4096) bounds activation memory: sequence layers run
  over row groups of at most that many tokens, two in flight.
- The library does no networking; fetch checkpoints with your own downloader
  and verify them against the checksums on the model cards.

An actor that owns the isolator:

```swift
actor VoiceIsolation {
    private let isolator: VoiceIsolator

    init(modelURL: URL) throws {
        isolator = try VoiceIsolator(contentsOf: modelURL)
    }

    func process(_ samples: [Float], sampleRate: Int) throws -> [Float] {
        try isolator.isolate(samples, sampleRate: sampleRate)
    }
}
```

## Command line

`xcodebuild -scheme voice-isolate-mlx -configuration Release` builds
`voice-isolate-mlx` into the derived data's `Build/Products/Release`:

```bash
voice-isolate-mlx noisy.wav clean.wav
voice-isolate-mlx noisy.wav clean.wav --mode streaming --lookahead 2
voice-isolate-mlx convert --src denoiser.safetensors --dst model_fp16.safetensors --fp16
```

Offline it writes the voice and the removed background (`clean_background.wav`)
as float32 WAV at 48 kHz; together they reconstruct the mono input. Without
`--model` it fetches `model_fp16` (or `model_fp32` with `--precision fp32`)
from the repositories above, verified by SHA-256, and falls back to the
released checkpoint (pinned revision, verified).

## Accuracy and speed

Apple M4 (10-core GPU, 16 GB):

| | fp32 | fp16 |
| --- | ---: | ---: |
| offline, one 8 s window | 1.66 s (RTF 0.21) | 1.34 s (RTF 0.17) |
| activations, 8 s window | +714 MiB | +430 MiB |
| live 20 ms push, median / p95 | 9.9 / 11.8 ms | 7.7 / 8.7 ms |
| output vs PyTorch fp32, max abs | 6e-8 to 1.1e-6 | 4e-5 to 2.4e-4 |
| voice SNR vs fp32, demo clips | reference | 63-78 dB |

fp16 denoises as fp32 does: SI-SDR against the clean recordings two demo clips
were made from matches fp32's to within 0.02 dB (19.93 and 28.22 dB; the noisy
inputs measure 6.0 and 9.0 dB).

The STFT runs on MLX's CPU stream (pocketfft, the FFT of PyTorch's CPU
backend) with the exact float32 bits of `torch.hann_window(2048)`, so the
spectrum is bit-identical to the reference's. On band-limited input (a 16 kHz
recording resampled to 48 kHz) the network amplifies float32 rounding about
1000-fold: there the reference differs from itself across PyTorch's CPU and
MPS backends by 1.0e-4, and this port by a similar amount.

The resampler is torchaudio's `kaiser_best`: 0.00 dB at 7 kHz and -145 dB
alias leakage for 48 to 16 kHz; 60 s takes 2-38 ms on the GPU depending on the
rate pair. Interpolating resamplers (linear, cubic) leave spectral images at
-25 to -32 dB above the source band when upsampling 16 kHz input; the model
does not remove them, since it only scales energy that is present.

## Tests

```bash
xcodebuild test -scheme VoiceIsolatorMLX-Package -destination 'platform=macOS'
```

runs the signal-processing tests. The model tests also run when given
checkpoints:

```bash
TEST_RUNNER_VOICE_ISOLATOR_MODEL=/path/VoiceIsolator_v1_48K_MLX/model_fp32.safetensors TEST_RUNNER_VOICE_ISOLATOR_REALTIME_MODEL=/path/VoiceIsolator_v1_Realtime_48K_MLX/model_fp32.safetensors xcodebuild test -scheme VoiceIsolatorMLX-Package -destination 'platform=macOS'
```

## Implementation

The 46 band tokens of every frame stay in one `[batch * bands * frames, 512]`
matrix; sequence layers address it through index maps, so the stream is never
transposed between the time and band axes. Custom Metal kernels
(`KernelSources.swift`) carry the axis changes and the hot paths: head split
with RoPE, head merge with exclusive self attention, the bidirectional
depthwise convolution, a selective scan that keeps the 16 states in
registers and computes its step size in-kernel, and GELU. The Mamba output
projections are folded at conversion, and the per-band encoders and decoder
heads run as batched matmuls over padded band weights.

## License

Apache-2.0 ([LICENSE](LICENSE)); see [NOTICE](NOTICE) for the upstream model
and code this port is derived from.
