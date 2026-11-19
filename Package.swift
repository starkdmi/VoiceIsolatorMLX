// swift-tools-version: 6.0
// VoiceIsolatorMLX: the Inworld Voice Isolator (48 kHz speech isolation) on
// MLX Swift - network, signal processing, checkpoint conversion and audio I/O.

import PackageDescription

let package = Package(
    name: "VoiceIsolatorMLX",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        .library(name: "VoiceIsolatorMLX", targets: ["VoiceIsolatorMLX"]),
        .executable(name: "voice-isolate-mlx", targets: ["voice-isolate-mlx"]),
    ],
    dependencies: [
        // 0.29.1 through 0.32.x are built and tested
        .package(url: "https://github.com/ml-explore/mlx-swift", "0.29.1" ..< "0.33.0")
    ],
    targets: [
        .target(
            name: "VoiceIsolatorMLX",
            dependencies: [.product(name: "MLX", package: "mlx-swift")]
        ),
        .executableTarget(
            name: "voice-isolate-mlx",
            dependencies: ["VoiceIsolatorMLX"]
        ),
        .testTarget(
            name: "VoiceIsolatorMLXTests",
            dependencies: ["VoiceIsolatorMLX"]
        ),
    ]
)
