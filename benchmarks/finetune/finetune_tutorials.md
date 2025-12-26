## Benchmarking instructions

This uses **relative paths** and requires you submit from **repo root**.
All sbatch scripts must include:

```bash
cd "$SLURM_SUBMIT_DIR"
test -d benchmarks || { echo "ERROR: submit from repo root"; exit 1; }
```

## Two stages only
- **Stage 1 (tune):** <=12 options, 2 seeds (0–1), eval episodes=3  
  **Budgets**: 200k (cheetah/walker/quadruped), 400k (humanoid)
- **Stage2 (ablation, final budgets):** top3 options, 12 seeds (0–11), eval episodes=10  
  **Budgets**: 1M (cheetah/walker/quadruped), 2M (humanoid)

## Algorithms
Run now:  
- Discrete (4): `sac, ppo, trpo, td3`  
- Continuous (4): `ct_sac, ct_td3, cppo, q_learning`  

---

## 0) One-time: create folders
```bash
mkdir -p benchmarks/finetune/manifests
mkdir -p benchmarks/hyperparams_gen
mkdir -p logs/finetune saved_models/finetune
```

---

## 1) Smoke test (tiny run to validate the pipeline)
```bash
python -u benchmarks/finetune/make_manifest.py   --stage stage1   --algos ct_sac   --env_ids cheetah-run   --hyperparams_dir benchmarks/hyperparams   --mode_prefix irregular_dt_option_   --out benchmarks/finetune/manifests/SMOKE_stage1_ctsac_cheetah.jsonl   --override_seeds 0   --override_timesteps_default 20000   --override_timesteps_humanoid 20000   --override_n_eval_episodes 2

python -u benchmarks/finetune/run_manifest_entry.py   --manifest benchmarks/finetune/manifests/SMOKE_stage1_ctsac_cheetah.jsonl   --index 0

find logs/finetune -name evaluations.npz | head
```

---

## 2) Stage 1 (tuning): <=12 options, 2 seeds, eval=3

### 2.1 Generate manifests (CT)
**Stampede3 (CPU)** — cheetah + walker:
```bash
python -u benchmarks/finetune/make_manifest.py   --stage stage1   --algos ct_sac,ct_td3,cppo,q_learning   --env_ids cheetah-run,walker-walk   --hyperparams_dir benchmarks/hyperparams   --mode_prefix irregular_dt_option_   --out benchmarks/finetune/manifests/stage1_ct_stampede3.jsonl
```

**Lonestar6 (CPU)** — quadruped:
```bash
python -u benchmarks/finetune/make_manifest.py   --stage stage1   --algos ct_sac,ct_td3,cppo,q_learning   --env_ids quadruped-run   --hyperparams_dir benchmarks/hyperparams   --mode_prefix irregular_dt_option_   --out benchmarks/finetune/manifests/stage1_ct_ls6.jsonl
```

**Vista (CPU or GPU)** — humanoid:
```bash
python -u benchmarks/finetune/make_manifest.py   --stage stage1   --algos ct_sac,ct_td3,cppo,q_learning   --env_ids humanoid-walk   --hyperparams_dir benchmarks/hyperparams   --mode_prefix irregular_dt_option_   --out benchmarks/finetune/manifests/stage1_ct_vista.jsonl
```

### 2.2 Generate manifests (Discrete)
Stampede3:
```bash
python -u benchmarks/finetune/make_manifest.py   --stage stage1   --algos sac,ppo,trpo,td3   --env_ids cheetah-run,walker-walk   --hyperparams_dir benchmarks/hyperparams   --mode_prefix irregular_dt_option_   --out benchmarks/finetune/manifests/stage1_discrete_stampede3.jsonl
```

LS6:
```bash
python -u benchmarks/finetune/make_manifest.py   --stage stage1   --algos sac,ppo,trpo,td3   --env_ids quadruped-run   --hyperparams_dir benchmarks/hyperparams   --mode_prefix irregular_dt_option_   --out benchmarks/finetune/manifests/stage1_discrete_ls6.jsonl
```

Vista:
```bash
python -u benchmarks/finetune/make_manifest.py   --stage stage1   --algos sac,ppo,trpo,td3   --env_ids humanoid-walk   --hyperparams_dir benchmarks/hyperparams   --mode_prefix irregular_dt_option_   --out benchmarks/finetune/manifests/stage1_discrete_vista.jsonl
```

### 2.3 Submit Stage 1 (CPU)
```bash
export TACC_PROJECT=YOUR_ALLOCATION

./benchmarks/finetune/submit_manifest.sh stampede3 benchmarks/finetune/manifests/stage1_ct_stampede3.jsonl 40
./benchmarks/finetune/submit_manifest.sh stampede3 benchmarks/finetune/manifests/stage1_discrete_stampede3.jsonl 40

./benchmarks/finetune/submit_manifest.sh ls6 benchmarks/finetune/manifests/stage1_ct_ls6.jsonl 20
./benchmarks/finetune/submit_manifest.sh ls6 benchmarks/finetune/manifests/stage1_discrete_ls6.jsonl 20

./benchmarks/finetune/submit_manifest.sh vista benchmarks/finetune/manifests/stage1_ct_vista.jsonl 20
./benchmarks/finetune/submit_manifest.sh vista benchmarks/finetune/manifests/stage1_discrete_vista.jsonl 20
```

### 2.4 Optional: Stage1 GPU submits
```bash
./benchmarks/finetune/submit_manifest.sh vista-gpu-gh benchmarks/finetune/manifests/stage1_ct_vista.jsonl 10
./benchmarks/finetune/submit_manifest.sh vista-gpu-gh benchmarks/finetune/manifests/stage1_discrete_vista.jsonl 10
```

---

## 3) Stage 1 -> Stage 2: select top3 modes (per algo, env)

```bash
mkdir -p benchmarks/hyperparams_gen/stage2_ct
mkdir -p benchmarks/hyperparams_gen/stage2_discrete

python -u benchmarks/finetune/score_and_select.py   --stage_log_root logs/finetune/stage1   --hyperparams_dir_in benchmarks/hyperparams   --hyperparams_dir_out benchmarks/hyperparams_gen/stage2_ct   --keep_topk 3   --metric auc

python -u benchmarks/finetune/score_and_select.py   --stage_log_root logs/finetune/stage1   --hyperparams_dir_in benchmarks/hyperparams   --hyperparams_dir_out benchmarks/hyperparams_gen/stage2_discrete   --keep_topk 3   --metric auc
```

---

## 4) Stage 2 (ablation): 3 options x 12 seeds, eval=10, FINAL budgets
Budgets are enforced by `stage_config.py`:
- 1M for cheetah/walker/quadruped
- 2M for humanoid

### 4.1 Generate Stage 2 manifests (CT)
Stampede3:
```bash
python -u benchmarks/finetune/make_manifest.py   --stage stage2   --algos ct_sac,ct_td3,cppo,q_learning   --env_ids cheetah-run,walker-walk   --hyperparams_dir benchmarks/hyperparams_gen/stage2_ct   --out benchmarks/finetune/manifests/stage2_ct_stampede3.jsonl
```

LS6:
```bash
python -u benchmarks/finetune/make_manifest.py   --stage stage2   --algos ct_sac,ct_td3,cppo,q_learning   --env_ids quadruped-run   --hyperparams_dir benchmarks/hyperparams_gen/stage2_ct   --out benchmarks/finetune/manifests/stage2_ct_ls6.jsonl
```

Vista:
```bash
python -u benchmarks/finetune/make_manifest.py   --stage stage2   --algos ct_sac,ct_td3,cppo,q_learning   --env_ids humanoid-walk   --hyperparams_dir benchmarks/hyperparams_gen/stage2_ct   --out benchmarks/finetune/manifests/stage2_ct_vista.jsonl
```

### 4.2 Generate Stage 2 manifests (Discrete)
Stampede3:
```bash
python -u benchmarks/finetune/make_manifest.py   --stage stage2   --algos sac,ppo,trpo,td3   --env_ids cheetah-run,walker-walk   --hyperparams_dir benchmarks/hyperparams_gen/stage2_discrete   --out benchmarks/finetune/manifests/stage2_discrete_stampede3.jsonl
```

LS6:
```bash
python -u benchmarks/finetune/make_manifest.py   --stage stage2   --algos sac,ppo,trpo,td3   --env_ids quadruped-run   --hyperparams_dir benchmarks/hyperparams_gen/stage2_discrete   --out benchmarks/finetune/manifests/stage2_discrete_ls6.jsonl
```

Vista:
```bash
python -u benchmarks/finetune/make_manifest.py   --stage stage2   --algos sac,ppo,trpo,td3   --env_ids humanoid-walk   --hyperparams_dir benchmarks/hyperparams_gen/stage2_discrete   --out benchmarks/finetune/manifests/stage2_discrete_vista.jsonl
```

### 4.3 Submit Stage 2 (CPU)
```bash
export TACC_PROJECT=YOUR_ALLOCATION

./benchmarks/finetune/submit_manifest.sh stampede3 benchmarks/finetune/manifests/stage2_ct_stampede3.jsonl 40
./benchmarks/finetune/submit_manifest.sh stampede3 benchmarks/finetune/manifests/stage2_discrete_stampede3.jsonl 40

./benchmarks/finetune/submit_manifest.sh ls6 benchmarks/finetune/manifests/stage2_ct_ls6.jsonl 20
./benchmarks/finetune/submit_manifest.sh ls6 benchmarks/finetune/manifests/stage2_discrete_ls6.jsonl 20

./benchmarks/finetune/submit_manifest.sh vista benchmarks/finetune/manifests/stage2_ct_vista.jsonl 20
./benchmarks/finetune/submit_manifest.sh vista benchmarks/finetune/manifests/stage2_discrete_vista.jsonl 20
```

### 4.4 Optional: Stage 2 GPU submits (recommended for humanoid)
```bash
./benchmarks/finetune/submit_manifest.sh vista-gpu-gh benchmarks/finetune/manifests/stage2_ct_vista.jsonl 10
./benchmarks/finetune/submit_manifest.sh vista-gpu-gh benchmarks/finetune/manifests/stage2_discrete_vista.jsonl 10
```

---

## 5) Monitoring on TACC (quick)
```bash
squeue -u "$USER"
squeue -u "$USER" -o "%.18i %.9P %.20j %.2t %.10M %.10l %.6D %R"

# Why pending / node assignment
scontrol show job <JOBID> | sed -n '1,160p'

# Accounting (after start/finish)
sacct -j <JOBID> --format=JobID,JobName%30,Partition,State,Elapsed,MaxRSS,AllocTRES%50

# Queue limits
qlimits

# Tail logs
tail -f slurm_<JOBID>_<ARRAYID>.out
```
