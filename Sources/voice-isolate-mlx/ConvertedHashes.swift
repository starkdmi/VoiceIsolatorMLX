// SHA-256 of the converted checkpoints as published on the Hugging Face Hub
// (written by tools/prepare_hf.py). A download that does not match is refused.

enum ConvertedHashes {
    static let offline: [String: String] = [
        "model_fp16.safetensors": "ffb1f72732c430cab90228306a90f2bdc714eaa6f6ebab90196c60a6c1ee1d28",
        "model_fp32.safetensors": "660c40d07e3837eb680dcd0c5ab8aa99bf4b16ea32af2383d00089cc442246ee",
    ]
    static let realtime: [String: String] = [
        "model_fp16.safetensors": "1a8d376e432319ff59be506b9f805287357d659d6a47d4cc51d46b68e0a4af11",
        "model_fp32.safetensors": "a72add45bcf0594e8bac631a62eca85d080677ac611612f26d2c898c58ee00a6",
    ]
}
