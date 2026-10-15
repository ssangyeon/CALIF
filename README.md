# Diagnosing and Correcting Reward Hacking in Verifiable Instruction Following

Official training code for [Diagnosing and Correcting Reward Hacking in Verifiable Instruction Following](https://arxiv.org/abs/2610.19525).

## Installation

```bash
git clone https://github.com/ssangyeon/CALIF.git
cd CALIF
# Install uv first if needed: https://docs.astral.sh/uv/
bash scripts/install.sh
source .venv/bin/activate
```

## Training

Settings are defined in [`configs/settings.json`](configs/settings.json). Run the command for your chosen model and method.

```bash
# Qwen3-1.7B, non-thinking
bash scripts/train/qwen3_1_7b_nonthinking_if_rlvr.sh
bash scripts/train/qwen3_1_7b_nonthinking_calif.sh

# Qwen3-1.7B, thinking
bash scripts/train/qwen3_1_7b_thinking_if_rlvr.sh
bash scripts/train/qwen3_1_7b_thinking_calif.sh

# Qwen3-4B, non-thinking
bash scripts/train/qwen3_4b_nonthinking_if_rlvr.sh
bash scripts/train/qwen3_4b_nonthinking_calif.sh

# Qwen3-4B, thinking
bash scripts/train/qwen3_4b_thinking_if_rlvr.sh
bash scripts/train/qwen3_4b_thinking_calif.sh

# Tulu 3 8B DPO, non-thinking
bash scripts/train/tulu3_8b_dpo_if_rlvr.sh
bash scripts/train/tulu3_8b_dpo_calif.sh
```

Launchers default to 8 GPUs. To change the GPU count (subject to available GPU memory):

```bash
bash scripts/train/qwen3_1_7b_nonthinking_calif.sh --gpus 4
```

Use `--help` for additional hardware options.

## Citation

```bibtex
@article{yoon2026diagnosing,
  title={Diagnosing and Correcting Reward Hacking in Verifiable Instruction Following},
  author={Yoon, Sangyeon and Seo, Wooseok and Ivison, Hamish and Hong, Hyesoo and Kim, Donghoon and Kang, Wonjun and No, Albert and Kim, Jaehyung},
  journal={arXiv preprint arXiv:2610.19525},
  year={2026}
}
```
