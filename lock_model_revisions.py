#!/usr/bin/env python3
"""Resolve model refs once per run before any repeat downloads weights."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import yaml
from benchmarks.benchmark_protocol import resolve_revision
from config_utils import load_config
from suite_config import suite_values


def lock_revisions(cfg, resolver=resolve_revision):
    locked, manifest, cache = deepcopy(cfg), {}, {}
    for suite in ('llm_infer', 'sd_infer', 'llm_train_real', 'llm_serve'):
        values = suite_values(cfg, suite) if suite == 'llm_serve' else (cfg.get(suite) or {})
        if not values.get('enabled', suite in ('llm_infer', 'sd_infer')) or not values.get('model'):
            continue
        if suite == 'llm_serve' and values.get('endpoint'):
            manifest[suite] = dict(verified=False, reason='external server revision cannot be attested locally')
            continue
        revision = values.get('revision') or 'main'
        filename = 'model_index.json' if suite == 'sd_infer' else 'config.json'
        key = (values['model'], revision, filename)
        if key not in cache:
            cache[key] = resolver(values['model'], revision, filename)
        resolved = cache[key]
        manifest[suite] = dict(model=values['model'], requested_revision=revision,
                               resolved_revision=resolved, verified=resolved is not None)
        if resolved:
            locked.setdefault(suite, {}).update(model=values['model'], revision=resolved)
    return locked, manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--manifest', required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    locked, manifest = lock_revisions(load_config(config_path))
    config_path.write_text(yaml.safe_dump(locked, sort_keys=False))
    Path(args.manifest).write_text(json.dumps(manifest, indent=2) + '\n')
    for suite, identity in manifest.items():
        print(f'[MODEL] {suite}: {identity.get("resolved_revision") or "unverified/local"}')


if __name__ == '__main__':
    main()
