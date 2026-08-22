# Self-supervised plannability v1

This package implements `SELF_SUPERVISED_PLANNABILITY_SPEC.md`.  SSP training
uses only a frozen LeWM, a fixed latent hit bit, and hard CEM.  It never
constructs or steps an environment.

The formal gate order is:

1. run repository tests and commit a clean source revision;
2. run `prepare --formal --full-dataset-hash` for all three tasks;
3. run the real-checkpoint short and four-pair full-budget `smoke`;
4. submit the nine create-only `train` roots;
5. after selection, run `profile` for each replicate;
6. only then run the secondary CLEAR v0.5 cells.

Every command refuses incompatible inputs or existing formal targets.  Smoke
artifacts explicitly record `formal_evidence=false` and cannot be promoted.
