"""Command line interface. Run python -m schnitz.cli or the installed schnitz command."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="SCHNITZELJAGD: Stigmergic Compactable Holographic Neural Indexed Trajectory Zettelkasten with Evolving Latents, Jointly Adapted by Gated Decoders")
    from . import __version__
    parser.add_argument("--version", action="version", version=f"SCHNITZELJAGD {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser('runs', help='Start, inspect or cooperatively stop a local detached curriculum')
    p.add_argument('action', choices=['start', 'status', 'stop', 'configure'])
    p.add_argument('--output', required=True)
    p.add_argument('--recipe')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--checkpoint-every', type=int)
    p.add_argument('--keep-checkpoints', type=int)
    p.add_argument('--no-archive', action='store_true')
    p = sub.add_parser('relocate-checkpoints', help='Verify and move stopped-stage checkpoints to external storage')
    p.add_argument('--run', required=True)
    p.add_argument('--destination', required=True)
    p.add_argument('--archive-hint')
    p = sub.add_parser('archive', help='Copy a verified checkpoint to an existing archive directory')
    p.add_argument('--run', required=True)
    p.add_argument('--destination', required=True)
    p.add_argument('--keep', type=int, default=3)
    p = sub.add_parser('restore', help='Restore a verified archive into a new run directory')
    p.add_argument('--archive', required=True)
    p.add_argument('--output', required=True)
    p = sub.add_parser('storage', help='Inspect filesystem capacity and run artifact sizes')
    p.add_argument('--path', required=True)
    p = sub.add_parser("launch", help="Prepare pinned data and execute a staged training curriculum")
    p.add_argument("--recipe", default="recipes/looped_starter_muon.yaml")
    p.add_argument("--output", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    p = sub.add_parser("datasets", help="List supported upstream teacher datasets")
    p = sub.add_parser("evaluate-teachers", help="Stored-only teacher NLL and payload interventions")
    p.add_argument("--run", required=True)
    p.add_argument("--episodes", required=True)
    p.add_argument("--output", help="Fresh result directory for this checkpoint")
    p.add_argument("--max-episodes", type=int, default=64)
    p.add_argument("--generate-tokens", type=int, default=0)
    p = sub.add_parser("doctor", help="Inspect runtime and test BF16 CUDA when available")
    p.add_argument("--require-spark", action="store_true")
    p = sub.add_parser("model-probe", help="Execute one-pass identity, recurrent causality and memory gradients")
    p.add_argument("--config", required=True)
    p.add_argument("--output")
    p = sub.add_parser("train", help="Train support/query episodes")
    p.add_argument("--config", help='Defaults to the committed checkpoint config with --resume')
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--init-from", help="Warm-start compatible weights with fresh optimizer/cache")
    p.add_argument("--stop-after", type=int, help="Checkpoint after this many new complete steps")
    p = sub.add_parser("evaluate", help="Freeze weights; serialize new memories and read stored-only")
    p.add_argument("--run", required=True)
    p.add_argument("--count", type=int)
    p.add_argument("--generate", action="store_true")
    p = sub.add_parser("evaluate-episodes", help="Stored-only answer NLL on general support/query JSONL")
    p.add_argument("--run", required=True)
    p.add_argument("--episodes", required=True)
    p = sub.add_parser("make-boolean", help="Balanced two-fact CPU composition diagnostic")
    p.add_argument("--output", required=True)
    p.add_argument("--worlds", type=int, default=128)
    p.add_argument("--split", default="boolean-train")
    p.add_argument("--operations", nargs="+", default=["xor"])
    p = sub.add_parser("make-multiuse", help="One write, many uses and variable-binding tasks")
    p.add_argument("--output", required=True)
    p.add_argument("--worlds", type=int, default=32)
    p.add_argument("--bindings", type=int, default=3)
    p.add_argument("--split", default="multiuse-train")
    p = sub.add_parser("evaluate-transfer", help="Write-once globally heterogeneous stored bank")
    p.add_argument("--run", required=True)
    p.add_argument("--episodes", required=True)
    p.add_argument("--compact", action="store_true")
    p.add_argument("--drop-supports", action="store_true")
    p.add_argument("--boolean-counterfactuals", action="store_true")
    p.add_argument("--binding-counterfactuals", action="store_true")
    p.add_argument("--persistent-compact", action="store_true")
    p = sub.add_parser("evaluate-depths", help="One frozen writer/bank, several native recurrent depths")
    p.add_argument("--run", required=True)
    p.add_argument("--episodes", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--depths", nargs="+", type=int, default=[1, 2, 3, 4])
    p.add_argument("--protocol", choices=["transfer", "teacher"], default="transfer")
    p.add_argument("--max-episodes", type=int, default=32)
    p = sub.add_parser("make-data", help="Create causally separated synthetic episodes")
    p.add_argument("--output", required=True)
    p.add_argument("--count", type=int, default=128)
    p.add_argument("--split", default="development")
    p.add_argument("--distractors", type=int, default=2)
    p = sub.add_parser("io-bench", help="Profile exact key scan and payload fetch; NOT a cold-I/O claim")
    p.add_argument("--path", required=True)
    p.add_argument("--records", type=int, default=1000)
    p.add_argument("--payload-dim", type=int, default=256)
    p.add_argument("--key-dim", type=int, default=64)
    p.add_argument("--reads", type=int, default=20)
    p.add_argument("--neighbors", type=int, default=16)
    p.add_argument("--cache-mib", type=int, default=32)
    p.add_argument("--output")
    p = sub.add_parser("compact-probe", help="Optimize compaction against a fixed random reader")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--reader", choices=["mlp", "attention"], default="mlp")
    p.add_argument("--output")
    args = parser.parse_args(argv)
    result = None
    if args.command == 'runs':
        from .operations import start_run, run_status, request_stop, configure_checkpoints
        if args.action == 'start':
            if not args.recipe:
                parser.error('runs start requires --recipe')
            result = start_run(args.recipe, args.output, resume=args.resume)
        elif args.action == 'stop':
            result = request_stop(args.output)
        elif args.action == 'configure':
            result = configure_checkpoints(args.output,
                checkpoint_every=args.checkpoint_every if args.checkpoint_every is not None else ...,
                keep_checkpoints=args.keep_checkpoints if args.keep_checkpoints is not None else ...,
                archive_dir=None if args.no_archive else ...)
        else:
            result = run_status(args.output)
    elif args.command == 'relocate-checkpoints':
        from .archiving import relocate_checkpoints
        result = relocate_checkpoints(args.run, args.destination, archive_hint=args.archive_hint)
    elif args.command == 'archive':
        from .archiving import archive_checkpoint
        from .checkpoints import resolve_checkpoint
        from .operations import file_lock
        if args.keep < 1:
            parser.error('Positive archive retention required')
        destination = Path(args.destination)
        if not destination.is_dir():
            raise FileNotFoundError('Archive destination must already exist on the intended disk')
        from .operations import run_lock
        with run_lock(args.run, clear_stop=False), file_lock(destination / '.archive.lock'):
            result = {'checkpoint': str(archive_checkpoint(resolve_checkpoint(args.run, verify=True),
                                        destination, keep=args.keep))}
    elif args.command == 'restore':
        from .archiving import restore_archive
        result = restore_archive(args.archive, args.output)
    elif args.command == 'storage':
        import shutil
        path = Path(args.path).resolve()
        usage = shutil.disk_usage(path)
        result = dict(path=str(path), filesystem_total_bytes=usage.total, filesystem_free_bytes=usage.free,
                      artifact_bytes=sum(p.stat().st_size for p in path.rglob('*') if p.is_file() and not p.is_symlink()),
                      pending_directories=[str(p) for p in path.rglob('.pending-*') if p.is_dir()],
                      note='Inspection only; no shared caches, runs or archives are deleted.')
    elif args.command == "launch":
        from .launch import launch
        result = launch(args.recipe, args.output, resume=args.resume, prepare_only=args.prepare_only)
    elif args.command == "datasets":
        from .trajectories import CATALOG
        result = CATALOG
    elif args.command == "evaluate-teachers":
        from .trajectory_eval import evaluate_teacher_run
        result = evaluate_teacher_run(args.run, args.episodes, max_episodes=args.max_episodes,
                                      generate_tokens=args.generate_tokens, output=args.output)
    elif args.command == "doctor":
        from .diagnostics import doctor
        result = doctor(args.require_spark)
    elif args.command == "model-probe":
        from .probes import model_probe
        result = model_probe(load_config(args.config))
    elif args.command == "train":
        from .training import train, config_from_run
        if not args.config and not args.resume:
            parser.error('Fresh training requires --config')
        config = load_config(args.config) if args.config else config_from_run(Path(args.output))
        if args.steps is not None:
            config.train.steps = args.steps
        result = train(config, args.output, resume=args.resume, stop_after=args.stop_after, init_from=args.init_from)
    elif args.command == "evaluate":
        from .training import evaluate_run
        if args.count is not None and args.count < 1:
            parser.error("--count must be positive")
        evaluate_run(args.run, count=args.count, generate=args.generate)
    elif args.command == "evaluate-episodes":
        from .training import evaluate_episode_file
        result = evaluate_episode_file(args.run, args.episodes)
    elif args.command == "evaluate-transfer":
        from .evaluation import evaluate_transfer_run
        full = evaluate_transfer_run(args.run, args.episodes, compact=args.compact,
                                     drop_supports=args.drop_supports, boolean_counterfactuals=args.boolean_counterfactuals,
                                     persistent_compact=args.persistent_compact,
                                     binding_counterfactuals=args.binding_counterfactuals)
        result = {key: value for key, value in full.items() if key != "rows"}
    elif args.command == "evaluate-depths":
        from .depth_eval import evaluate_depths
        result = evaluate_depths(args.run, args.episodes, args.output, depths=tuple(args.depths),
                                 protocol=args.protocol, max_episodes=args.max_episodes)
    elif args.command == "make-boolean":
        from .data import make_boolean_world, save_episodes
        path = Path(args.output)
        if args.worlds < 1 or path.exists():
            raise ValueError("Positive count and new output path required")
        episodes = [e for i in range(args.worlds) for e in make_boolean_world(i, split=args.split, operations=tuple(args.operations))]
        save_episodes(path, episodes)
        result = {"output": str(path), "worlds": args.worlds, "queries": len(episodes)}
    elif args.command == "make-multiuse":
        from .data import make_multiuse_world, save_episodes
        if args.worlds < 1 or args.bindings < 1:
            parser.error("Positive world/binding counts required")
        path = Path(args.output)
        if path.exists():
            raise FileExistsError(path)
        episodes = [e for i in range(args.worlds) for e in make_multiuse_world(i, split=args.split, bindings=args.bindings)]
        save_episodes(path, episodes)
        result = {"output": str(path), "worlds": args.worlds, "queries": len(episodes)}
    elif args.command == "make-data":
        from .data import make_episode, save_episodes
        if args.count < 1 or args.distractors < 0:
            parser.error("Invalid episode counts")
        path = Path(args.output)
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_episodes(path, [make_episode(i, split=args.split, distractors=args.distractors)
                             for i in range(args.count)])
        result = {"output": str(path), "count": args.count}
    elif args.command == "io-bench":
        from .diagnostics import io_benchmark
        result = io_benchmark(args.path, records=args.records, payload_dim=args.payload_dim,
                              key_dim=args.key_dim, reads=args.reads, neighbors=args.neighbors,
                              cache_mib=args.cache_mib)
    elif args.command == "compact-probe":
        from .diagnostics import compact_probe
        result = compact_probe(steps=args.steps, seed=args.seed, reader_kind=args.reader)
    if result is not None:
        text = json.dumps(result, indent=2) + "\n"
        if getattr(args, "output", None) and args.command in {"io-bench", "compact-probe", "model-probe"}:
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        print(text, end="")


if __name__ == "__main__":
    main()
