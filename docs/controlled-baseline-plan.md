# Controlled baseline implementation plan

Approved scope: selectable single-GPU baseline, workload identity, synchronized timing.

- Add `--baseline` orchestration with one selected physical GPU, five repeats, single-worker inference, and world-size-one training. Keep the existing default configuration available.
- Record architecture, deterministic seed, resolved model/tokenizer revisions, prompt hashes, scheduler settings, and timing protocol. Comparison keys separate incompatible work; missing identity prevents strict classification.
- Warm the measured shape, synchronize device work, and coordinate replicated worker readiness. Use a common monotonic start and latest completion for aggregate throughput. Abort barriers on worker failures.
- Add CPU-only regression tests for configuration, comparison identities, timing coordination and failure handling. Run unit tests, Python compilation, and shell syntax validation.

No GPU benchmark will be claimed validated on this macOS host. Historical data is unchanged; power measurement and real-model training methodology remain separate work.
