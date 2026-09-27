"""Frozen-feature global routing capacity at several address widths."""
import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import random
import signal

from safetensors.torch import load_file
import torch

from probe_global_routing import global_features, address_scores, assess
from probe_routing_features import pair_loss
from probe_routing_stop import StopProbe
from schnitz.checkpoints import resolve_checkpoint
from schnitz.data import load_episodes
from schnitz.operations import atomic_json, control_dir, run_lock, stop_requested
from schnitz.optimizers import optimizer_report
from schnitz.probe_state import restore_probe_state, save_probe_state, parameter_names
from schnitz.runtime import configure_memory, available_host_memory, memory_metrics, compute_watchdog
from schnitz.tracking import Tracking
from schnitz.training import config_from_run, autocast_context
from schnitz.trajectories import file_sha256


def widen_weights(weights, width, generator):
    """Keep the original block; tiny key-only residuals let added dimensions learn.

    New query rows start at zero, new map blocks at identity. Key residuals alter
    cosine normalization slightly, so initial function equality is measured rather
    than claimed. An all-zero extension would leave the new subspace dead.
    """
    names = {'key.weight', 'address.weight', 'query_map.weight', 'query_head.weight'}
    if set(weights) != names:
        raise ValueError('Expected the four independent address matrices')
    original, hidden = weights['key.weight'].shape
    if (width < original or weights['query_head.weight'].shape != (original, hidden) or
            any(weights[n].shape != (original, original) for n in ('address.weight', 'query_map.weight'))):
        raise ValueError('Width must preserve the original compatible address block')
    if width == original:
        return {k:v.clone() for k,v in weights.items()}
    result = {}
    for name in ('key.weight','query_head.weight'):
        value = torch.zeros(width, hidden, dtype=weights[name].dtype)
        value[:original] = weights[name]
        if name == 'key.weight':
            value[original:] = torch.randn(width-original,hidden,generator=generator) * weights[name].std() * .001
        result[name] = value
    for name in ('address.weight','query_map.weight'):
        value = torch.zeros(width,width,dtype=weights[name].dtype)
        value[:original,:original] = weights[name]
        value[original:,original:] = torch.eye(width-original,dtype=weights[name].dtype)
        result[name] = value
    return result


def run(source, features_root, output, widths=(64,128,256), steps=3200, batch_size=128):
    if steps < 1 or batch_size < 1 or not widths or len(set(widths)) != len(widths):
        raise ValueError('Positive budgets and distinct widths required')
    output.mkdir(parents=True,exist_ok=True)
    with run_lock(output,clear_stop=False):
        checkpoint = resolve_checkpoint(source,verify=True)
        inputs = json.loads((features_root/'inputs.json').read_text())
        endpoint = features_root/'full_state_query-resume.pt'
        prior = torch.load(endpoint,weights_only=True,map_location='cpu')
        if (prior['identity'] != inputs or prior['step'] != inputs['steps'] or
                inputs['checkpoint_manifest_sha256'] != file_sha256(checkpoint/'manifest.json')):
            raise ValueError('Frozen feature/source identity differs')
        original_width = prior['model']['key.weight'].shape[0]
        if any(w < original_width for w in widths):
            raise ValueError('Narrowing requires a different initialization experiment')
        config = copy.deepcopy(config_from_run(checkpoint))
        config.train.optimizer = 'muon'
        config.train.steps, config.train.seed = steps,67
        config.train.gradient_accumulation = 1
        config.train.loop_counts = []
        config.train.wandb_group = 'global-routing-width-probe'
        config.train.train_worlds = inputs['train_worlds']
        config.train.episodes_file = str(features_root/'train.jsonl')
        config.model.freeze_backbone = True
        configure_memory(config.train)
        torch.set_num_threads(config.train.threads)
        torch.set_float32_matmul_precision('highest')
        data,episodes,ids,hashes = {},{},{},{}
        for split in ('train','heldout'):
            episode_path = features_root/f'{split}.jsonl'
            if file_sha256(episode_path) != inputs[f'{split}_sha256']:
                raise ValueError('Frozen episode corpus changed')
            episodes[split] = load_episodes(episode_path)
            path = features_root/f'{split}-features.safetensors'
            values,ids[split] = global_features(load_file(str(path)),episodes[split])
            data[split] = {k:v.to(config.train.device) for k,v in values.items()}
            hashes[split] = file_sha256(path)
        if set(ids['train']) & set(ids['heldout']):
            raise ValueError('Training and heldout sources overlap')
        common = {'checkpoint_manifest_sha256':inputs['checkpoint_manifest_sha256'],
                  'source_endpoint_sha256':file_sha256(endpoint),'source_identity':inputs,
                  'features_sha256':hashes,'config':asdict(config),'steps':steps,'batch_size':batch_size,
                  'seed':67,'widths':list(widths),'script_sha256':file_sha256(__file__),
                  'initialization':'Preserved old block; zero query extension, key noise at 0.001 source weight std, identity map extension',
                  'cosine_precision':'writer BF16 normalization then FP32/highest cosine',
                  'notice':'Unequal parameter/key-byte budgets; feature-only diagnostic, no downstream capability claim.'}
        if (output/'inputs.json').exists() and json.loads((output/'inputs.json').read_text()) != common:
            raise ValueError('Width experiment identity changed')
        atomic_json(output/'inputs.json',common)
        reference = StopProbe(prior['model'],False).to(config.train.device)
        with torch.no_grad(),autocast_context(config):
            initial_scores = address_scores(reference,data['heldout'],torch.arange(len(episodes['heldout']),device=config.train.device),'global')
        del reference
        for width in widths:
            root = output/f'width-{width}'
            root.mkdir(exist_ok=True)
            random.seed(67)
            torch.manual_seed(67)
            weights = widen_weights(prior['model'],width,torch.Generator().manual_seed(71))
            model = StopProbe(weights,False).to(config.train.device)
            identity = common | {'routing_width':width,'parameters':sum(p.numel() for p in model.parameters()),
                                 'serialized_key_bytes_per_record':width*4}
            optimizer = torch.optim.Muon(model.parameters(),lr=config.train.learning_rate,
                momentum=config.train.muon_momentum,ns_steps=config.train.muon_ns_steps,
                weight_decay=config.train.weight_decay,adjust_lr_fn='match_rms_adamw')
            sampler = torch.Generator().manual_seed(67)
            path = root/'resume.pt'
            start = restore_probe_state(path,model,optimizer,sampler,identity)
            def save(step):
                save_probe_state(path,model,optimizer,sampler,identity,step,reserve_bytes=config.train.min_free_disk_bytes)
            atomic_json(root/'optimizer.json',{'groups':optimizer_report(optimizer),'parameter_names':parameter_names(model,optimizer)})
            if not path.exists():
                save(0)
                with torch.no_grad(),autocast_context(config):
                    scores = address_scores(model,data['heldout'],torch.arange(len(episodes['heldout']),device=config.train.device),'global')
                eligible = torch.isfinite(initial_scores)
                if not torch.equal(torch.isfinite(scores), eligible):
                    raise ValueError('Width initialization changed score eligibility')
                atomic_json(root/'initial.json',{'max_abs_score_change':(scores[eligible]-initial_scores[eligible]).abs().max().item(),
                    'same_top_two_queries':int((scores.argsort(descending=True,stable=True)[:,:2] == initial_scores.argsort(descending=True,stable=True)[:,:2]).all(-1).sum()),
                    'heldout':assess(model,data['heldout'],episodes['heldout'],ids['heldout'],config)})
            with Tracking(config,root) as tracking:
                if tracking.run is not None:
                    tracking.run.config.update({'width_probe':identity})
                for step in range(start,steps):
                    available = available_host_memory()
                    if stop_requested(output) or (available is not None and available < config.train.min_system_available_bytes):
                        save(step)
                        raise RuntimeError('Width probe stopped at complete optimizer boundary')
                    indices = torch.randint(len(episodes['train']),(batch_size,),generator=sampler).to(config.train.device)
                    with compute_watchdog(config.train.stall_timeout_seconds, device=config.train.device):
                        optimizer.zero_grad(set_to_none=True)
                        with autocast_context(config):
                            scores = address_scores(model,data['train'],indices,'global')
                            loss = pair_loss(scores,data['train']['required'][indices],data['train']['lengths'][indices])
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(),config.train.clip_grad_norm,error_if_nonfinite=True)
                        optimizer.step()
                    if (step+1)%20 == 0:
                        row = {'step':step+1,'loss':loss.item(),'routing_width':width,'resume_attempt':tracking.attempt,
                               **memory_metrics(config.train.device)}
                        with (root/'metrics.jsonl').open('a') as handle:
                            handle.write(json.dumps(row)+'\n')
                        tracking.log(row)
                        print(json.dumps(row),flush=True)
                save(steps)
            for split in ('train','heldout'):
                atomic_json(root/f'{split}.json',assess(model,data[split],episodes[split],ids[split],config))
            atomic_json(root/'completed.json',{'identity':identity,'step':steps})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source','features','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--widths',nargs='+',type=int,default=[64,128,256])
    parser.add_argument('--steps',type=int,default=3200)
    parser.add_argument('--batch-size',type=int,default=128)
    args = parser.parse_args()
    def stop(_signal,_frame):
        (control_dir(args.output)/'STOP').touch()
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    run(args.source,args.features,args.output,args.widths,args.steps,args.batch_size)
