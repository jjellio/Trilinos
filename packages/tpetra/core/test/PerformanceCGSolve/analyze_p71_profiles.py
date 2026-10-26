#!/usr/bin/env python3
"""P7.1 independent-launch summary and GPU probe distributions (stdlib only).

Input is the output directory of run_p71_completion_profile.sh. All GPU
clock64 samples are approximate same-thread intervals, not network latency.
"""
from __future__ import annotations
import argparse
import csv
from collections import defaultdict
import math
from pathlib import Path
import re
import statistics
import sys


def read_tsv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def emit_csv(path, data, fields):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(data)


def median(values):
    return statistics.median(values) if values else float('nan')


def percentile(values, p):
    if not values:
        return float('nan')
    a = sorted(values)
    position = p * (len(a) - 1)
    lower = int(position)
    upper = min(lower + 1, len(a) - 1)
    return a[lower] + (a[upper] - a[lower]) * (position - lower)


def metric_line(text, section, method):
    needle = f"[NVSHMEM P6.2 {section}]"
    found = []
    for line in text.splitlines():
        if needle not in line:
            continue
        m = re.search(r"(?:^|\s)method=([^\s]+)", line)
        if m and m.group(1) == method:
            found.append(line)
    if len(found) != 1:
        raise ValueError(f"{section}/{method}: expected one line, got {len(found)}")
    m = re.search(r"(?:^|\s)median=([0-9.]+)", found[0])
    if not m:
        raise ValueError(f"{section}/{method}: missing median")
    return float(m.group(1))


def load_one(entry, folder, ranks):
    row = dict(entry)
    if entry['status'] != '0':
        raise ValueError(f"MPI launch {entry['run_id']} failed with {entry['status']}")
    log_path = Path(entry['log'])
    if not log_path.exists():
        raise ValueError(f"Missing {log_path}")
    text = log_path.read_text(errors='replace')
    if '[NVSHMEM P6.2] SUMMARY status=PASS' not in text:
        raise ValueError(f"Missing P6.2 PASS: {log_path}")
    if not any('[NVSHMEM P7.1 PROFILE]' in l and 'status=PASS' in l for l in text.splitlines()):
        raise ValueError(f"Missing P7.1 PROFILE PASS: {log_path}")
    for label,section,method in [
        ('mpi_import_us','DIRECT','tpetra_doImport'),
        ('p62_import_us','DIRECT','descriptor_two_stream'),
        ('mpi_including_start_barrier_us','P71_FULL','tpetra_barrier_plus_import'),
        ('p62_including_start_barrier_us','P71_FULL','p62_barrier_plus_import'),
        ('p71_producer_kernel_event_us','P71_PROFILE','producer_cuda_kernel_event'),
        ('p71_receiver_kernel_event_us','P71_PROFILE','receiver_cuda_kernel_event'),
        ('p71_host_device_wait_us','P71_PROFILE','cudaDeviceSynchronize_host_wait'),
        ('p71_probed_gpu_import_host_us','P71_PROFILE','instrumented_gpu_import_host'),
        ('p71_post_import_readback_us','P71_PROFILE','post_import_probe_readback'),
    ]:
        row[label] = metric_line(text, section, method)
    # Do not equate separate rank-maximum stage medians with an end-to-end sum.
    # Already-computed FULL samples are correctly rank paired by the app.
    row['p62_vs_mpi_import_only'] = row['p62_import_us'] / row['mpi_import_us']
    row['p62_full_vs_mpi_import_only'] = row['p62_including_start_barrier_us'] / row['mpi_import_us']
    prefix = entry['output_prefix']
    # The same P7.1 instrumentation can now be run by the dedicated
    # matrix-free Import-topology executable (case=topology). Preserve the
    # historical case=matrix behavior for the MiniFE benchmark.
    match = re.search(r'\[NVSHMEM P6\.2 SETUP\] case=([^\s]+)', text)
    case = match.group(1) if match is not None else 'matrix'
    if not re.fullmatch(r'[A-Za-z0-9_-]+', case):
        raise ValueError('Invalid case token in SETUP line')
    raw_path = Path(prefix + '_' + case + '_p62_samples.csv')
    if raw_path.exists():
        raw = read_csv(raw_path)
        expected = {'DIRECT': {'descriptor_two_stream','tpetra_doImport'},
                    'P71_FULL': {'p62_barrier_plus_import','tpetra_barrier_plus_import'},
                    'P71_PROFILE': {'producer_cuda_kernel_event','receiver_cuda_kernel_event'}}
        checked = 0
        for section, methods in expected.items():
            for meth in methods:
                grouped = defaultdict(list)
                for x in raw:
                    if x['section'] == section and x['method'] == meth:
                        grouped[int(x['iteration'])].append((int(x['rank']),float(x['us'])))
                if not grouped:
                    raise ValueError(f"Missing {section}/{meth} in {raw_path}")
                maxima = []
                for records in grouped.values():
                    if len(records) != ranks or {r for r,_ in records} != set(range(ranks)):
                        raise ValueError(f"Malformed rank group {section}/{meth}")
                    maxima.append(max(v for _,v in records))
                reported = metric_line(text, section, meth)
                if abs(median(maxima)-reported)>0.05:
                    raise ValueError(f"Log vs CSV median mismatch {section}/{meth}")
                checked += 1
        row['rank_csv_checks'] = checked
    else:
        raise ValueError(f"Missing raw rank CSV {raw_path}")
    probes = []
    for r in range(ranks):
        path = Path(f"{prefix}_{case}_p71_rank{r}_probes.csv")
        if not path.exists():
            raise ValueError(f"Missing per-rank GPU probe data {path}")
        content = read_csv(path)
        for x in content:
            if int(x['rank']) != r:
                raise ValueError(f"Probe filename/content rank mismatch: {path}")
            if int(x['clock_khz']) <= 0:
                raise ValueError(f"GPU probe clock rate invalid in {path}")
            if x['kind'] == 'signal_poll_and_scatter' and int(x['poll_attempts']) < 1:
                raise ValueError(f"Receiver failed to record signal test in {path}")
            probes.append({**x, 'run_id':entry['run_id'], 'transport':entry['transport'],
                           'variant':entry['variant'],'chunk':entry['chunk'],
                           'topology':entry['topology'], 'pass':entry['pass'],
                           **{k:entry[k] for k in ('volume','degree','intra_degree',
                               'payload_bytes','source','target') if k in entry}})
    row['sampled_gpu_work_units'] = len(probes)
    row['gpu_poll_samples'] = sum(x['kind']=='signal_poll_and_scatter' for x in probes)
    return row, probes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory', type=Path)
    ap.add_argument('--output-stem',default='',
                    help='Optional filename prefix to avoid overwriting another analyzer')
    args = ap.parse_args()
    folder=args.directory.resolve()
    stem=(args.output_stem + '_' if args.output_stem else '')
    entries=read_tsv(folder/'manifest.tsv')
    if not entries:
        ap.error('Empty manifest')
    config=(folder/'config.txt').read_text()
    nmatch=re.search(r'^np=(\d+)$',config,re.M)
    if nmatch is None:
        ap.error('Missing np in config.txt')
    ranks=int(nmatch.group(1))
    loaded=[]; probes=[]; problems=[]
    for e in entries:
        try:
            row, raw = load_one(e,folder,ranks)
            loaded.append(row); probes.extend(raw)
        except (ValueError, OSError, KeyError, ZeroDivisionError) as exc:
            problems.append(f"{e['run_id']}: {exc}")
    metrics = [
        'mpi_import_us','p62_import_us','mpi_including_start_barrier_us',
        'p62_including_start_barrier_us','p71_producer_kernel_event_us',
        'p71_receiver_kernel_event_us','p71_host_device_wait_us',
        'p71_probed_gpu_import_host_us','p71_post_import_readback_us',
        'p62_vs_mpi_import_only','p62_full_vs_mpi_import_only',
        'rank_csv_checks','sampled_gpu_work_units','gpu_poll_samples']
    fields=list(entries[0])+metrics
    emit_csv(folder/(stem+'runs.csv'), loaded, fields)
    summary=[]
    by_config=defaultdict(list)
    config_keys=['topology','transport','variant','chunk']
    config_keys += [key for key in ('volume','degree','intra_degree','payload_bytes',
                     'source','target') if key in entries[0]]
    for row in loaded:
        by_config[tuple(row[key] for key in config_keys)].append(row)
    for k, group in sorted(by_config.items()):
        item=dict(zip(config_keys,k))
        item['independent_launches']=len(group)
        for m in metrics:
            item[m+'_median']=median([float(row[m]) for row in group])
        summary.append(item)
    emit_csv(folder/(stem+'summary.csv'), summary, config_keys+['independent_launches']+
             [m+'_median' for m in metrics])
    probe_by=defaultdict(list)
    probe_keys=config_keys+['route','kind']
    for row in probes:
        probe_by[tuple(row[k] for k in probe_keys)].append(row)
    probed=[]
    for key, group in sorted(probe_by.items()):
        item=dict(zip(probe_keys,key))
        item['sample_count']=len(group)
        item['mpi_launches']=len(set(x['run_id'] for x in group))
        # Pooling these observations describes probe distributions, not
        # independent-run confidence intervals or transport latencies.
        for term in ['first_us','second_us','poll_attempts']:
            values=[float(x[term]) for x in group]
            item[term+'_p10']=percentile(values,0.1)
            item[term+'_p50']=median(values)
            item[term+'_p90']=percentile(values,0.9)
        probed.append(item)
    emit_csv(folder/(stem+'probe_distributions.csv'), probed,
             probe_keys+['sample_count','mpi_launches']+
             [m+'_'+q for m in ('first_us','second_us','poll_attempts') for q in ('p10','p50','p90')])
    report=['# P7.1 completed Import and GPU phase probes', '',
            f'Independent MPI launches: {len(loaded)} PASS / {len(entries)} requested.',
            '', '## Metric cautions', '',
            '- `DIRECT` rows use the original **unmodified, uninstrumented** P6.2 kernel path.',
            '- The `P71_PROFILE` and GPU probes are obtained from separate instrumented-clone kernels; they are *not* a valid uninstrumented performance result.',
            '- `first_us/second_us` mean gather/PUT-call-return for sends, polling/scatter for receives, and local-copy/not-applicable for local values.',
            '- PUT-call return is **not** remote PUT completion; polling duration includes producer and receiver scheduling and does not isolate NIC latency.',
            '- Same-thread `clock64()` and `clockRate` yield approximate SM intervals, not synchronized inter-SM timestamps. Event durations may include scheduling/interference.',
            '- Samples are selected at fixed descriptor strides; first and last descriptors/peers need not have equal selection probabilities.',
            '- Separate GPU phase medians cannot be added to reconstruct end-to-end critical path. Use Nsight Systems for overlap.',
            '- `P71_FULL` is per-rank (barrier + operation), then per-iteration max across ranks; do not add independent barrier/Import medians.',
            '', '## Complete-Import by configuration', '',
            '| Topology | Transport | Variant | K | MPI (µs) | P6.2 (µs) | P6.2 + barrier (µs) | P6.2 full / MPI | independent launches |',
            '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for row in summary:
        report.append('| {} | {} | {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {} |'.format(
            (row['topology'] + (' d='+str(row['degree'])+' b='+str(row['payload_bytes'])+
                 ' src='+str(row['source']) if 'degree' in row else '')),
            row['transport'],row['variant'],row['chunk'],
            row['mpi_import_us_median'],row['p62_import_us_median'],
            row['p62_including_start_barrier_us_median'],
            row['p62_full_vs_mpi_import_only_median'],row['independent_launches']))
    report.extend(['','## Probe distributions','',
                   f'`{stem}probe_distributions.csv` contains sampled per-descriptor p10/p50/p90 by inter/intra/local routing.',
                   'Compare independent-launch medians in `summary.csv`, not raw samples as independent trials.', ''])
    if problems:
        report.extend(['## Problems', '']+['- '+p for p in problems]+[''])
    (folder/(stem+'analysis.md')).write_text('\n'.join(report)+'\n')
    print(f'P7.1 analyzer: {len(loaded)} valid launches / {len(entries)} planned; '
          f'{len(probes)} GPU probe records; errors={len(problems)}')
    for p in problems: print('FAILED',p,file=sys.stderr)
    return 1 if problems else 0

if __name__ == '__main__':
    sys.exit(main())
