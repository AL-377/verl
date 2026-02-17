#!/usr/bin/env bash
set -xeuo pipefail

# ============================================================
# 离线评估脚本：对已保存的 checkpoint 逐个跑 validation
# 用法: bash eval_checkpoints.sh
# ============================================================

exp_name='riddle-llama-8b-ins-ds-on-policy-add-coef-seq-mean-token-mean'
CKPTS_DIR=/mnt/hdfs/ljt_save/models/${exp_name}
TEST_FILE=/opt/tiger/verl/data/eval_verl_compat_gpqa_mathv2.parquet
TRAIN_FILE=/opt/tiger/verl/data/train_verl_compat_add_think.parquet

# 你的原始模型路径（step 0 用）
MODEL_PATH=/mnt/hdfs/ljt_save/models/Llama-31-8b-ins/0e9e39f249a16976918f6564b8830bc894c89659

# 要评估的 step 列表：从 0 开始每隔 5 step
# 根据你训到 44 step、save_freq=5，checkpoint 有: 5, 10, 15, 20, 25, 30, 35, 40
STEPS=(0 5 10 15 20 25 30 35 40)

# 评估用的参数
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 20))
sp_size=4
gen_tp=1
offload=True
use_dynamic_bsz=True
temperature=1.0
val_top_p=0.7
top_k=-1

infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))

for step in "${STEPS[@]}"; do
    echo "=========================================="
    echo "Evaluating step ${step}"
    echo "=========================================="

    if [ "$step" -eq 0 ]; then
        # Step 0: 用原始模型，不 resume
        EVAL_MODEL_PATH="${MODEL_PATH}"
        RESUME_MODE="disable"
        RESUME_PATH=""
    else
        # 用 checkpoint 里的 hf_model
        HF_PATH="${CKPTS_DIR}/global_step_${step}/actor/huggingface"
        if [ ! -d "${HF_PATH}" ]; then
            echo "WARNING: ${HF_PATH} not found, skipping step ${step}"
            continue
        fi
        EVAL_MODEL_PATH="${HF_PATH}"
        RESUME_MODE="disable"
        RESUME_PATH=""
    fi

    python3 -m recipe.dapo.main_dapo \
        data.train_files="${TRAIN_FILE}" \
        data.val_files="${TEST_FILE}" \
        data.val_batch_size=128 \
        data.prompt_key=prompt \
        data.truncation='left' \
        data.max_prompt_length=${max_prompt_length} \
        data.max_response_length=${max_response_length} \
        data.gen_batch_size=16 \
        data.train_batch_size=16 \
        actor_rollout_ref.rollout.n=16 \
        algorithm.adv_estimator=grpo \
        algorithm.use_kl_in_reward=False \
        algorithm.kl_ctrl.kl_coef=0.0 \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.kl_loss_coef=0.0 \
        actor_rollout_ref.actor.clip_ratio_low=0.2 \
        actor_rollout_ref.actor.clip_ratio_high=0.28 \
        actor_rollout_ref.actor.clip_ratio_c=3.0 \
        algorithm.filter_groups.enable=False \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${infer_ppo_max_token_len} \
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
        actor_rollout_ref.model.path="${EVAL_MODEL_PATH}" \
        actor_rollout_ref.model.enable_gradient_checkpointing=False \
        actor_rollout_ref.actor.optim.lr=5e-7 \
        actor_rollout_ref.actor.ppo_mini_batch_size=16 \
        actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.grad_clip=1.0 \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
        actor_rollout_ref.rollout.enforce_eager=True \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
        actor_rollout_ref.rollout.enable_chunked_prefill=True \
        actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
        actor_rollout_ref.rollout.temperature=${temperature} \
        actor_rollout_ref.rollout.top_p=1.0 \
        actor_rollout_ref.rollout.top_k="${top_k}" \
        actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
        actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
        actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.rollout.val_kwargs.n=1 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
        actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
        actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
        reward_model.reward_manager=dapo \
        +reward.reward_kwargs.overlong_buffer_cfg.enable=True \
        +reward.reward_kwargs.overlong_buffer_cfg.len=$((1024 * 4)) \
        +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
        +reward.reward_kwargs.overlong_buffer_cfg.log=True \
        +reward.reward_kwargs.max_resp_len=${max_response_length} \
        actor_rollout_ref.rollout.max_model_len=$((max_prompt_length + max_response_length)) \
        trainer.logger='["console","swanlab"]' \
        trainer.project_name="riddle-llama-eval" \
        trainer.experiment_name="${exp_name}-eval-step-${step}" \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=1 \
        trainer.val_before_train=True \
        trainer.val_only=True \
        trainer.test_freq=9999 \
        trainer.save_freq=9999 \
        trainer.total_epochs=1 \
        trainer.resume_mode="${RESUME_MODE}" \
        trainer.default_local_dir="${CKPTS_DIR}/eval_tmp"

    echo "Step ${step} evaluation done."
done

echo "=========================================="
echo "All evaluations complete!"
echo "=========================================="

