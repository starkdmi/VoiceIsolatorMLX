import AVFoundation
import Foundation

/// Audio file reading (any format AVFoundation decodes) and WAV writing.
/// Channels are kept; `Audio.mono` averages them, as the reference does.
public enum AudioIO {
    /// Decoded audio: planar float32 channels and the sample rate.
    public struct Audio {
        public var channels: [[Float]]
        public var sampleRate: Int
        public var mono: [Float] { toMono(channels) }
    }

    /// Reads a file as planar float32 (integer PCM scaled by its full range).
    ///
    /// `AVAudioFile.read(into:)` may return fewer frames than asked for
    /// (compressed formats stop at packet boundaries), so the file is read in
    /// blocks until it is exhausted.
    public static func read(_ url: URL) throws -> Audio {
        let file = try AVAudioFile(forReading: url, commonFormat: .pcmFormatFloat32,
                                   interleaved: false)
        let format = file.processingFormat
        let channelCount = Int(format.channelCount)
        let block: AVAudioFrameCount = 1 << 16
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: block) else {
            throw VoiceIsolatorError.audio("cannot allocate a buffer for \(url.path)")
        }
        var channels = [[Float]](repeating: [], count: channelCount)
        for c in 0 ..< channelCount { channels[c].reserveCapacity(Int(max(file.length, 0))) }
        while file.framePosition < file.length {
            try file.read(into: buffer, frameCount: block)
            let count = Int(buffer.frameLength)
            if count == 0 { break }
            for c in 0 ..< channelCount {
                channels[c].append(
                    contentsOf: UnsafeBufferPointer(start: buffer.floatChannelData![c], count: count))
            }
        }
        return Audio(channels: channels, sampleRate: Int(format.sampleRate))
    }

    public enum Subtype { case float32, pcm16 }

    /// Writes a mono WAV. PCM16 output above full scale is scaled down to peak
    /// at 1.0 instead of being clipped.
    public static func writeWAV(_ url: URL, samples: [Float], sampleRate: Int,
                                subtype: Subtype = .float32) throws {
        var data = Data()
        func u32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
        func u16(_ v: UInt16) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
        let bytesPerSample = subtype == .float32 ? 4 : 2
        let payload = samples.count * bytesPerSample
        data.append(contentsOf: Array("RIFF".utf8))
        u32(UInt32(36 + payload + (subtype == .float32 ? 14 : 0)))
        data.append(contentsOf: Array("WAVE".utf8))
        data.append(contentsOf: Array("fmt ".utf8))
        u32(subtype == .float32 ? 18 : 16)
        u16(subtype == .float32 ? 3 : 1)
        u16(1)
        u32(UInt32(sampleRate))
        u32(UInt32(sampleRate * bytesPerSample))
        u16(UInt16(bytesPerSample))
        u16(UInt16(8 * bytesPerSample))
        if subtype == .float32 {
            u16(0)
            data.append(contentsOf: Array("fact".utf8))
            u32(4)
            u32(UInt32(samples.count))
        }
        data.append(contentsOf: Array("data".utf8))
        u32(UInt32(payload))
        switch subtype {
        case .float32:
            for s in samples { u32(s.bitPattern) }
        case .pcm16:
            let peak = samples.reduce(Float(0)) { max($0, abs($1)) }
            let scale: Float = peak > 1 ? 1 / peak : 1
            for s in samples {
                let v = max(-32768, min(32767, (s * scale * 32768).rounded()))
                u16(UInt16(bitPattern: Int16(v)))
            }
        }
        try data.write(to: url)
    }
}
