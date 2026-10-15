# Upstream attribution

The `verl/` package and packaging configuration are based on
[verl-project/verl](https://github.com/verl-project/verl), commit
`9f73954a87e247de4c31bc2f5222969e395b2904`.

The vendored training runtime includes CALIF integration for constraint-free
reference likelihoods, cached reference responses, final-answer masks, and verifier rewards.
Use this pinned runtime rather than installing an arbitrary newer verl release.

`if_rlvr/ifeval_oi/` contains instruction-verification code adapted from Google
Research IFEval and AllenAI open-instruct. Original copyright and Apache-2.0
notices are retained. See `LICENSE` and `Notice.txt`.

Model weights and datasets are separate artifacts and retain their respective
licenses and access conditions. They are not distributed in this Git repository.
