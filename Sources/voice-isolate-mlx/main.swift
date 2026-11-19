import Foundation
import MLX
import VoiceIsolatorMLX

let usage = """
    voice-isolate-mlx: Inworld Voice Isolator on MLX (48 kHz speech isolation)

    USAGE
      voice-isolate-mlx <input> <output> [options]
          Isolates the voice. Offline, the removed background is written beside
          the output (<output>_background.wav); both are float32 WAV, together
          they reconstruct the aligned mono input. Streaming writes the voice.
        --model PATH          converted MLX file or released .safetensors
                              (default: fetched for --mode and --precision)
        --precision P         fp16 (default), fp32 or bf16; quantized files
                              compute in their own precision
        --mode offline|streaming
        --lookahead N         streaming look-ahead in 10 ms frames (0, 2, 4)
        --output-sr N         output sample rate (default 48000, 0 = input rate)
        --background PATH     background output path
        --no-background       write only the voice
        --float               voice-only output as float32 instead of PCM16
        --chunk-seconds S     offline render window (default: checkpoint's, 8)
        --overlap-seconds S   offline window overlap (default: checkpoint's, 2)
        --cache-limit-mb N    MLX buffer cache bound (default 512; larger is
                              slightly faster, 0 gives the smallest footprint)

      voice-isolate-mlx convert --src PATH --dst PATH [--q8 [--group-size 32] | --fp16 | --bf16]
          Converts a released checkpoint to the MLX layout: float32, 8-bit
          linear layers (float32 compute), or float16 / bfloat16 compute.

      voice-isolate-mlx convert-all --out DIR
          Fetches both released checkpoints and writes float32, float16 and
          8-bit files.
    """

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data("error: \(message)\n".utf8))
    exit(1)
}

func parsed<T>(_ value: T?, _ message: String) -> T {
    guard let value else { fail(message) }
    return value
}

nonisolated(unsafe) var args = Array(CommandLine.arguments.dropFirst())
if args.isEmpty || args.contains("-h") || args.contains("--help") {
    print(usage)
    exit(args.isEmpty ? 1 : 0)
}

func option(_ name: String) -> String? {
    guard let i = args.firstIndex(of: name) else { return nil }
    guard i + 1 < args.count else { fail("\(name) needs a value") }
    let value = args[i + 1]
    args.removeSubrange(i ... i + 1)
    return value
}

func flag(_ name: String) -> Bool {
    guard let i = args.firstIndex(of: name) else { return false }
    args.remove(at: i)
    return true
}

do {
    switch args.first {
    case "convert":
        args.removeFirst()
        guard let src = option("--src"), let dst = option("--dst") else {
            fail("convert needs --src and --dst")
        }
        let groupSize = Int(option("--group-size") ?? "32") ?? 32
        let q8 = flag("--q8")
        let dtype: DType? = flag("--fp16") ? .float16 : (flag("--bf16") ? .bfloat16 : nil)
        try Weights.convert(src: URL(fileURLWithPath: src), dst: URL(fileURLWithPath: dst),
                            quantization: q8 ? Quantization(bits: 8, groupSize: groupSize) : nil,
                            dtype: dtype)
        print("\(dst) written")
    case "convert-all":
        args.removeFirst()
        let out = URL(fileURLWithPath: option("--out") ?? "models")
        try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
        for (mode, name) in [(VoiceIsolator.Mode.offline, "voice-isolator-v1"),
                             (.streaming, "voice-isolator-v1-realtime")] {
            let src = try await Hub.original(mode)
            let variants: [(String, Quantization?, DType?)] = [
                ("", nil, nil), ("-fp16", nil, .float16), ("-q8", Quantization(), nil),
            ]
            for (suffix, quantization, dtype) in variants {
                let dst = out.appendingPathComponent("\(name)-mlx\(suffix).safetensors")
                try Weights.convert(src: src, dst: dst, quantization: quantization, dtype: dtype)
                print("\(dst.path) written")
            }
        }
    default:
        let modelPath = option("--model")
        // GPU.set(cacheLimit:) exists in every mlx-swift release the package
        // accepts (0.29 through 0.32); Memory.cacheLimit only from 0.30
        GPU.set(cacheLimit:
            parsed(Int(option("--cache-limit-mb") ?? "512"), "--cache-limit-mb is a number") << 20)
        let precision = option("--precision") ?? "fp16"
        let dtype: DType
        switch precision {
        case "fp32": dtype = .float32
        case "fp16": dtype = .float16
        case "bf16": dtype = .bfloat16
        default: fail("--precision is fp32, fp16 or bf16")
        }
        let mode = parsed(VoiceIsolator.Mode(rawValue: option("--mode") ?? "offline"),
                          "--mode is offline or streaming")
        let lookahead = option("--lookahead").map { parsed(Int($0), "--lookahead is a number") }
        let outputSR = Int(option("--output-sr") ?? "48000") ?? 48000
        let backgroundPath = option("--background")
        let noBackground = flag("--no-background")
        let asFloat = flag("--float")
        let chunk = option("--chunk-seconds").map { parsed(Double($0), "bad --chunk-seconds") }
        let overlap = option("--overlap-seconds").map {
            parsed(Double($0), "bad --overlap-seconds")
        }
        guard args.count == 2 else { fail("expected <input> <output>\n\n\(usage)") }
        let input = URL(fileURLWithPath: args[0])
        let output = URL(fileURLWithPath: args[1])

        let modelURL: URL
        if let modelPath {
            modelURL = URL(fileURLWithPath: modelPath)
        } else {
            modelURL = try await Hub.checkpoint(mode, precision: precision == "fp32" ? "fp32" : "fp16")
        }
        let network = try Network(contentsOf: modelURL, dtype: dtype)
        let isolator = try VoiceIsolator(network: network, mode: mode, lookaheadFrames: lookahead)
        let audio = try AudioIO.read(input)
        let mono = audio.mono
        let outRate = outputSR == 0 ? audio.sampleRate : outputSR
        let start = Date()
        let voice = try isolator.isolate(mono, sampleRate: audio.sampleRate,
                                         outputSampleRate: outRate, chunkSeconds: chunk,
                                         overlapSeconds: overlap)
        let elapsed = Date().timeIntervalSince(start)
        let writeBackground = mode == .offline && !noBackground
        try AudioIO.writeWAV(output, samples: voice, sampleRate: outRate,
                             subtype: (writeBackground || asFloat) ? .float32 : .pcm16)
        print(String(format: "%@: %.2fs in %.2fs", output.path,
                     Double(voice.count) / Double(outRate), elapsed))
        if writeBackground {
            var reference = mono
            if outRate != audio.sampleRate {
                let r = DSP.fitLength(
                    DSP.resample(MLXArray(mono, [mono.count]), from: audio.sampleRate, to: outRate),
                    voice.count)
                eval(r)
                reference = r.asArray(Float.self)
            }
            let background = zip(reference, voice).map { $0 - $1 }
            let bgURL = backgroundPath.map { URL(fileURLWithPath: $0) }
                ?? URL(fileURLWithPath: output.deletingPathExtension().path + "_background.wav")
            try AudioIO.writeWAV(bgURL, samples: background, sampleRate: outRate, subtype: .float32)
            print("\(bgURL.path): the removed background")
        }
    }
} catch {
    fail("\(error)")
}
