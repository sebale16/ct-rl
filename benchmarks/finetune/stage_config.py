# benchmarks/finetune/stage_config.py

# Two-stage pipeline:
# - stage1: tuning (<=12 options), 3 seeds, eval=3
# - stage2: ablation (top3 options), 12 seeds, eval=10 with FINAL budgets

STAGES = {
    "stage1": {
        "seeds": [0, 1, 2],
        "n_eval_episodes": 3,
        "timesteps_default": 200_000,
        "timesteps_overrides": {
            "humanoid-walk": 400_000,
        },
    },
    "stage2": {
        "seeds": list(range(12)),  # 0..11
        "n_eval_episodes": 10,
        "timesteps_default": 1_000_000,  # cheetah/walker/quadruped
        "timesteps_overrides": {
            "humanoid-walk": 2_000_000,
        },
    },
}
