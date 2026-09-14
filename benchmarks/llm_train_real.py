#!/usr/bin/env python3
"""Single GPU causal-LM training on deterministic synthetic token sequences."""
import argparse
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path

from benchmark_protocol import SEED, resolve_revision, runtime_gpu_name


def supervised_tokens(batch_size, seq_len):
    if batch_size < 1 or seq_len < 2:
        raise ValueError('Training needs batch_size >= 1 and seq_len >= 2')
    return batch_size * (seq_len - 1)


def training_step(model, optimizer, tokens, torch, *, autocast_context=nullcontext,
                  scaler=None, diagnostics=None, max_overflow_retries=8):
    diagnostics = diagnostics if diagnostics is not None else {}
    for attempt in range(max_overflow_retries + 1):
        diagnostics['stage'] = 'zero_grad'
        optimizer.zero_grad(set_to_none=True)
        diagnostics['stage'] = 'forward_loss'
        # AutoModelForCausalLM shifts logits and labels internally exactly once.
        with autocast_context():
            output = model(input_ids=tokens, labels=tokens.clone())
            loss = output.loss.float()
        if not torch.isfinite(loss).all():
            raise FloatingPointError('Nonfinite training loss')
        diagnostics['stage'] = 'backward'
        (scaler.scale(loss) if scaler is not None else loss).backward()
        if scaler is not None:
            scaler.unscale_(optimizer)
        diagnostics['stage'] = 'gradient_check'
        finite = all(torch.isfinite(p.grad).all() for p in model.parameters()
                     if p.grad is not None)
        if not finite:
            if scaler is None:
                raise FloatingPointError('Nonfinite training gradient')
            # GradScaler skips this update and reduces its scale. Retry the same
            # batch so requested steps count actual updates, including retry cost.
            scaler.step(optimizer)
            scaler.update()
            diagnostics['overflow_retries'] = diagnostics.get('overflow_retries', 0) + 1
            if attempt == max_overflow_retries:
                raise FloatingPointError('Nonfinite training gradient after loss-scale retries')
            continue
        diagnostics['stage'] = 'optimizer_step'
        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
            scaler.update()
        return loss


def run(cfg):
    metric = {'benchmark_schema_version': 2, 'suite': 'llm_train_real',
              'model': cfg.get('model'), 'status': 'skipped'}
    if not cfg.get('enabled', False):
        return {**metric, 'reason': 'disabled'}
    if int(os.environ.get('WORLD_SIZE', cfg.get('world_size', 1))) != 1:
        return {**metric, 'reason': 'real-model training currently supports one GPU'}
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        return {**metric, 'reason': f'optional dependency unavailable: {exc}'}
    if not torch.cuda.is_available():
        return {**metric, 'reason': 'GPU runtime unavailable'}
    from energy import EnergySampler
    backend = 'amd' if getattr(torch.version, 'hip', None) else 'nvidia'
    metric.update(gpu_backend=backend, gpu_count=1, world_size=1)
    sampler = None
    diagnostics = {'phase': 'setup', 'stage': 'configuration', 'overflow_retries': 0}
    try:
        batch_size, seq_len = int(cfg['batch_size']), int(cfg['seq_len'])
        steps, warmup = int(cfg['steps']), int(cfg.get('warmup_steps', 2))
        tokens_per_step = supervised_tokens(batch_size, seq_len)
        if steps < 1 or warmup < 0:
            raise ValueError('steps must be positive and warmup_steps nonnegative')
        seed = int(cfg.get('seed', SEED))
        dtype_name = cfg.get('dtype', 'fp16')
        dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16,
                 'fp32': torch.float32}[dtype_name]
        metric.update(dtype=dtype_name, training_precision_protocol='fp32_parameters_autocast_v1',
                      parameter_dtype='fp32', gradient_scaling=dtype_name == 'fp16',
                      max_overflow_retries=8)
        torch.manual_seed(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        device = torch.device('cuda:0')
        name = cfg['model']
        revision = resolve_revision(name, cfg.get('revision'))
        diagnostics['stage'] = 'model_load'
        model = AutoModelForCausalLM.from_pretrained(
            name, revision=revision, torch_dtype=torch.float32,
            trust_remote_code=False).to(device).train()
        autocast_context = lambda: torch.autocast(
            device_type='cuda', dtype=dtype, enabled=dtype_name != 'fp32')
        scaler = torch.amp.GradScaler('cuda') if dtype_name == 'fp16' else None
        model_revision = getattr(model.config, '_commit_hash', None) or revision
        learning_rate = float(cfg.get('learning_rate', 3e-4))
        weight_decay = float(cfg.get('weight_decay', 0.01))
        epsilon = float(cfg.get('adam_epsilon', 1e-6))
        optim = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay, eps=epsilon)
        vocab_size = model.config.vocab_size
        generator = torch.Generator(device='cpu').manual_seed(seed)
        # Fixed synthetic batch generated and transferred before warmup.
        tokens = torch.randint(vocab_size, (batch_size, seq_len),
                               generator=generator, dtype=torch.long).to(device)
        diagnostics['phase'] = 'warmup'
        for step in range(warmup):
            diagnostics['step'] = step + 1
            training_step(model, optim, tokens, torch, autocast_context=autocast_context,
                          scaler=scaler, diagnostics=diagnostics)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        sampler = EnergySampler(backend, device_indices=[0])
        sampler.start()
        started = time.perf_counter()
        diagnostics['phase'] = 'measurement'
        for step in range(steps):
            diagnostics['step'] = step + 1
            loss = training_step(model, optim, tokens, torch,
                                 autocast_context=autocast_context,
                                 scaler=scaler, diagnostics=diagnostics)
        torch.cuda.synchronize(device)
        finished = time.perf_counter()
        elapsed = finished - started
        energy = sampler.stop(started_s=started, ended_s=finished)
        sampler = None
        metric.update(
            status='ok', timing_method='synchronized_training_compute_v3',
            data_protocol='fixed_seeded_random_tokens_resident_v1',
            objective='causal_next_token_internal_shift', seed=seed,
            model_source='local_path' if Path(name).is_dir() else 'huggingface',
            model_revision=model_revision, tokenizer_revision=None,
            tokenizer_used=False, model_family=model.config.model_type,
            param_count=sum(p.numel() for p in model.parameters()),
            vocab_size=vocab_size, dtype=dtype_name, seq_len=seq_len,
            batch_size=batch_size, steps=steps, warmup_steps=warmup,
            tokens_per_step=tokens_per_step, tokens_per_sec=steps*tokens_per_step/elapsed,
            steps_per_sec=steps/elapsed, time_s=elapsed, final_loss=float(loss.detach()),
            correctness_passed=True, optimizer='AdamW', learning_rate=learning_rate,
            weight_decay=weight_decay, adam_epsilon=epsilon, adam_betas=[0.9, 0.999],
            allow_tf32=False, gpu_name=runtime_gpu_name(backend, 0, torch.cuda.get_device_name(device)),
            framework_gpu_name=torch.cuda.get_device_name(device),
            peak_memory_bytes=torch.cuda.max_memory_allocated(device),
            **energy)
        if metric.get('energy_j'):
            metric['tokens_per_joule'] = steps*tokens_per_step/metric['energy_j']
    except Exception as exc:
        metric.update(status='failed', error=str(exc), failure_phase=diagnostics['phase'],
                      failure_stage=diagnostics['stage'], failure_step=diagnostics.get('step'))
    finally:
        if sampler is not None:
            sampler.stop()
    metric['overflow_retries'] = diagnostics['overflow_retries']
    return metric


def main():
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    with open(args.config) as handle:
        cfg = (yaml.safe_load(handle) or {}).get('llm_train_real', {})
    metric = run(cfg)
    Path('results').mkdir(exist_ok=True)
    with open('results/metrics.jsonl', 'a') as handle:
        handle.write(json.dumps(metric, allow_nan=False) + '\n')
    print(json.dumps(metric, indent=2, allow_nan=False))
    return 1 if metric['status'] == 'failed' else 0


if __name__ == '__main__':
    raise SystemExit(main())
