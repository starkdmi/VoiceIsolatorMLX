import CryptoKit
import Foundation
import VoiceIsolatorMLX

/// Checkpoint resolution for the command line: the converted MLX checkpoints
/// (`model_fp32` / `model_fp16`) when published, otherwise the released
/// PyTorch checkpoint (pinned revision, SHA-256 verified), converted in memory.
/// The library itself does no networking.
enum Hub {
    struct Converted {
        let repo: String
        let files: [String: String]  // file name -> SHA-256
    }

    static let converted: [VoiceIsolator.Mode: Converted] = [
        .offline: Converted(repo: "starkdmi/VoiceIsolator_v1_48K_MLX", files: ConvertedHashes.offline),
        .streaming: Converted(
            repo: "starkdmi/VoiceIsolator_v1_Realtime_48K_MLX", files: ConvertedHashes.realtime),
    ]

    static let originalRepo = "inworld/voice-isolator"
    static let originalRevision = "9b0285b3fe8403aa8e2f1dbb2165a80a4294c970"
    static let originals: [VoiceIsolator.Mode: (file: String, sha256: String)] = [
        .offline: ("denoiser.safetensors",
                   "647335a6aeb4dc706b895532c209d51e0e84a4849d17b339f00d4392a050ae53"),
        .streaming: ("denoiser_realtime.safetensors",
                     "76598f05523c0669e76562a8b2f31c9d395fca316e80c907ee11775e7497ae0f"),
    ]

    static func sha256(_ url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while let chunk = try handle.read(upToCount: 1 << 24), !chunk.isEmpty {
            hasher.update(data: chunk)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    static var cacheDirectory: URL {
        FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("voice-isolator-mlx")
    }

    /// Downloads `remote` to `local` and verifies it; nil when the file is not
    /// on the Hub (HTTP 404).
    static func fetch(_ remote: URL, to local: URL, sha256 digest: String?) async throws -> URL? {
        let fm = FileManager.default
        if fm.fileExists(atPath: local.path) { return local }
        let (tmp, response) = try await URLSession.shared.download(from: remote)
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        if status == 404 || status == 401 {
            try? fm.removeItem(at: tmp)
            return nil
        }
        guard status == 200 else {
            throw VoiceIsolatorError.download("HTTP \(status) for \(remote)")
        }
        if let digest {
            let got = try sha256(tmp)
            guard got == digest else {
                try? fm.removeItem(at: tmp)
                throw VoiceIsolatorError.download(
                    "checksum mismatch for \(remote.lastPathComponent): \(got)")
            }
        }
        try fm.createDirectory(at: local.deletingLastPathComponent(),
                               withIntermediateDirectories: true)
        try? fm.removeItem(at: local)
        try fm.moveItem(at: tmp, to: local)
        return local
    }

    /// The released PyTorch checkpoint for `mode`.
    static func original(_ mode: VoiceIsolator.Mode) async throws -> URL {
        let (file, digest) = originals[mode]!
        let hfCache = URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent(
            ".cache/huggingface/hub/models--inworld--voice-isolator/snapshots/\(originalRevision)/\(file)")
        if FileManager.default.fileExists(atPath: hfCache.path) { return hfCache }
        let remote = URL(
            string: "https://huggingface.co/\(originalRepo)/resolve/\(originalRevision)/\(file)")!
        let local = cacheDirectory.appendingPathComponent("\(originalRevision)/\(file)")
        guard let url = try await fetch(remote, to: local, sha256: digest) else {
            throw VoiceIsolatorError.download("\(originalRepo)/\(file) is not available")
        }
        return url
    }

    /// The checkpoint for `mode` at `precision` ("fp32" or "fp16"): the
    /// converted file when published, else the released one.
    static func checkpoint(_ mode: VoiceIsolator.Mode, precision: String) async throws -> URL {
        let fm = FileManager.default
        if let c = converted[mode] {
            let file = "model_\(precision == "fp32" ? "fp32" : "fp16").safetensors"
            let hfCache = URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent(
                ".cache/huggingface/hub/models--\(c.repo.replacingOccurrences(of: "/", with: "--"))/snapshots")
            if let snapshots = try? fm.contentsOfDirectory(atPath: hfCache.path) {
                for snapshot in snapshots {
                    let candidate = hfCache.appendingPathComponent("\(snapshot)/\(file)")
                    if fm.fileExists(atPath: candidate.path) { return candidate }
                }
            }
            let remote = URL(string: "https://huggingface.co/\(c.repo)/resolve/main/\(file)")!
            let local = cacheDirectory.appendingPathComponent("\(c.repo)/\(file)")
            if let url = try await fetch(remote, to: local, sha256: c.files[file]) { return url }
        }
        return try await original(mode)
    }
}
