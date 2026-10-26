#!/usr/bin/env python3
"""Analyze independent P7.1b matrix-free MPI/NVSHMEM benchmark launches.

The metric of a run is the median over iterations of the maximum rank's
COMPLETED Import time. Treat MPI *launches*, not epochs, as replications.
Also compute correctly rank-paired barrier-plus-P6.2 elapsed time.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import re
import statistics
import sys


def median(values):
    return statistics.median(values) if values else float('nan')


def load_table(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def save_table(path, fields, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def resolve_run_file(folder, recorded_path):
    """Prefer files from the analyzed results directory over stale site paths.

    Sweep manifests contain absolute paths from the originating cluster. A
    tarball extraction preserves only their basenames. This keeps old manifests
    usable without rewriting them and also supports in-place cluster runs.
    """
    recorded = Path(recorded_path)
    local = folder / recorded.name
    if local.is_file():
        return local
    if recorded.is_file():
        return recorded
    raise ValueError(f'missing run artifact: {recorded.name}')


def check_pass_markers(text):
    """Validate *all* completion stages produced by the matrix-free driver.

    NvshmemP62::benchmarkCase emits P6.2 CASE, not the standalone P6.2
    top-level SUMMARY (only NvshmemP62::run prints that marker).
    """
    required = (
        '[NVSHMEM TOPOLOGY IMPORTER] status=PASS',
        '[NVSHMEM P6.2 CASE] case=topology status=PASS',
        '[NVSHMEM TOPOLOGY] SUMMARY status=PASS',
    )
    absent = [marker for marker in required if marker not in text]
    if absent:
        raise ValueError('missing Importer, P6.2 case or topology correctness PASS marker: '
                         + ', '.join(absent))


def direct_sample_tables(path, n_ranks):
    groups = defaultdict(dict)
    for row in load_table(path):
        if row['case'] != 'topology':
            raise ValueError('wrong case in raw timing file')
        key = (row['section'], row['method'])
        it = int(row['iteration'])
        rank = int(row['rank'])
        if not 0 <= rank < n_ranks or it < 0:
            raise ValueError('bad rank or iteration in raw file')
        if rank in groups[key].get(it, {}):
            raise ValueError('duplicate rank timing record')
        groups[key].setdefault(it, {})[rank] = float(row['us'])
    for key, epochs in groups.items():
        for epoch, entries in epochs.items():
            if set(entries) != set(range(n_ranks)):
                raise ValueError(f'incomplete per-rank epoch for {key} epoch {epoch}')
    return groups


def rankmax_medians(tables, key):
    if key not in tables or not tables[key]:
        raise ValueError(f'missing metric {key}')
    return median([max(v.values()) for _, v in sorted(tables[key].items())])


def paired_barrier_plus_import(tables, n_ranks):
    ready = tables.get(('DIRECT','descriptor_two_stream'))
    barrier = tables.get(('BARRIER','outside_gpu_import_timing'))
    if ready is None or barrier is None or set(ready) != set(barrier):
        raise ValueError('P6.2 direct/barrier epochs missing or not matched')
    return median([
        max(ready[epoch][rank] + barrier[epoch][rank] for rank in range(n_ranks))
        for epoch in sorted(ready)
    ])


def check_traffic(path, n_ranks, expected_degree, expected_cross, expected_ppn):
    traffic = load_table(path)
    if len(traffic) != n_ranks*n_ranks:
        raise ValueError('traffic CSV must contain exactly NxN directed pairs')
    seen = set()
    sends = [0] * n_ranks
    receives = [0] * n_ranks
    send_degrees = [0] * n_ranks
    receive_degrees = [0] * n_ranks
    cross = [0] * n_ranks
    hosts = defaultdict(set)
    for row in traffic:
        sender, receiver = int(row['sender_rank']), int(row['receiver_rank'])
        if not (0 <= sender < n_ranks and 0 <= receiver < n_ranks):
            raise ValueError('invalid rank in topology traffic file')
        if (sender,receiver) in seen:
            raise ValueError('duplicate traffic edge')
        seen.add((sender,receiver))
        data = int(row['doubles'])
        if data < 0 or int(row['bytes']) != data*8:
            raise ValueError('traffic size is not in doubles / 8-byte values')
        hosts[row['sender_node']].add(sender)
        if data:
            if sender == receiver:
                raise ValueError('self communication edge')
            sends[sender] += data
            receives[receiver] += data
            send_degrees[sender] += 1
            receive_degrees[receiver] += 1
            if row['sender_node'] != row['receiver_node']:
                cross[sender] += data
    if set(send_degrees) != {expected_degree} or set(receive_degrees) != {expected_degree}:
        raise ValueError('actual peer degree does not match requested degree')
    if len(set(sends)) != 1 or len(set(receives)) != 1 or sends != receives:
        raise ValueError('communication volume not balanced')
    if any(x != expected_cross for x in cross):
        raise ValueError('cross-node byte counts differ from expected metadata')
    if any(len(ranks) != expected_ppn for ranks in hosts.values()):
        raise ValueError('observed hosts do not have expected PPN')
    return sends[0], sum(cross), len(hosts)


def parse_run(row, folder, n_ranks, ppn):
    if row['status'] != '0':
        raise ValueError(f'nonzero MPI exit code/status={row["status"]}')
    prefix = Path(row['output_prefix'])
    log = resolve_run_file(folder, row['log'])
    if log.name != prefix.name + '.log':
        raise ValueError('manifest log name disagrees with output prefix')
    text = log.read_text(errors='replace')
    check_pass_markers(text)
    match = re.search(r'\[NVSHMEM TOPOLOGY\] nranks=(\d+) nodes=(\d+) ppn=(\d+)\s+' \
                      r'offDegree=(\d+) intraDegree=(\d+)\s+[^\n]*?' \
                      r'crossNodeDoublesPerRank=(\d+)', text)
    if match is None:
        raise ValueError('missing topology metadata in log')
    ranks, nodes, logged_ppn, off, intra, cross = map(int, match.groups())
    if ranks != n_ranks or logged_ppn != ppn or off != int(row['degree']) or \
       intra != int(row['intra_degree']):
        raise ValueError('rank count, PPN or degree metadata mismatch')
    remote, total_cross, nodes_csv = check_traffic(
        resolve_run_file(folder, str(prefix)+'_traffic.csv'),
        n_ranks, off+intra, cross, ppn)
    if nodes_csv != nodes:
        raise ValueError('node count mismatch')
    tables = direct_sample_tables(
        resolve_run_file(folder, str(prefix)+'_topology_p62_samples.csv'), n_ranks)
    mpi = rankmax_medians(tables, ('DIRECT','tpetra_doImport'))
    p62 = rankmax_medians(tables, ('DIRECT','descriptor_two_stream'))
    full = paired_barrier_plus_import(tables, n_ranks)
    if mpi <= 0 or p62 <= 0:
        raise ValueError('invalid zero timing')
    result = dict(row)
    result.update(mpi_import_us=mpi, p62_import_us=p62,
                  p62_including_barrier_us=full,
                  speedup_mpi_over_p62=mpi/p62,
                  speedup_mpi_over_p62_full=mpi/full,
                  remote_doubles_per_pe=remote,
                  cross_node_doubles_global=total_cross,
                  distinct_nodes=nodes,
                  status='PASS')
    # p71 independently records paired full samples. Do not assert identical
    # reduction with direct-only, since they represent different sections.
    if row['mode'] == 'p71':
        result['p71_matched_full_us'] = rankmax_medians(
            tables, ('P71_FULL','p62_barrier_plus_import'))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory', type=Path)
    args = ap.parse_args()
    folder = args.directory.resolve()
    manifest = folder / 'manifest.tsv'
    if not manifest.exists():
        ap.error(f'Missing {manifest}')
    runs = list(csv.DictReader(manifest.open(newline=''),delimiter='\t'))
    config = (folder/'config.txt').read_text()
    n_ranks = int(re.search(r'^np=(\d+)$',config,re.M).group(1))
    ppn = int(re.search(r'^ppn=(\d+)$',config,re.M).group(1))
    parsed, errors = [], []
    for row in runs:
        try:
            parsed.append(parse_run(row,folder,n_ranks,ppn))
        except (ValueError, KeyError, OSError, AttributeError) as exc:
            errors.append(f'run_id={row.get("run_id")} {exc}')
    fields = list(runs[0]) + [
        'mpi_import_us','p62_import_us','p62_including_barrier_us',
        'speedup_mpi_over_p62','speedup_mpi_over_p62_full',
        'remote_doubles_per_pe','cross_node_doubles_global','distinct_nodes','p71_matched_full_us'] if runs else []
    save_table(folder/'runs.csv', fields, parsed)
    grouped = defaultdict(list)
    grouping = ['topology','volume','degree','intra_degree','payload_bytes',
                'source','target','variant','chunk','transport','mode']
    for row in parsed:
        grouped[tuple(row[k] for k in grouping)].append(row)
    stats = ['mpi_import_us','p62_import_us','p62_including_barrier_us',
             'speedup_mpi_over_p62','speedup_mpi_over_p62_full']
    summary = []
    for key, group in sorted(grouped.items()):
        item = dict(zip(grouping,key))
        item['independent_mpi_launches'] = len(group)
        for metric in stats:
            item[metric+'_median'] = median([float(r[metric]) for r in group])
        summary.append(item)
    save_table(folder/'summary.csv',grouping+['independent_mpi_launches']+
               [m+'_median' for m in stats],summary)
    error_report = folder/'analysis_errors.txt'
    if errors:
        error_report.write_text('\n'.join(errors)+'\n')
        for error in errors: print('ERROR',error,file=sys.stderr)
    elif error_report.exists():
        error_report.unlink()  # clear obsolete reports from a previous parse
    print(f'[P7.1b ANALYSIS] total_launches={len(runs)} PASS={len(parsed)} failed={len(errors)}')
    print('The unit of independent replication is an MPI launch, NOT a timed epoch.')
    for item in summary:
        print('  {} degree={} intra={} bytes={} source={} {} k={} transport={} launches={} '
              'MPI={:.3f}us P62={:.3f}us P62+barrier={:.3f}us'.format(
                  item['topology'],item['degree'],item['intra_degree'],item['payload_bytes'],
                  item['source'],item['variant'],item['chunk'],item['transport'],
                  item['independent_mpi_launches'], item['mpi_import_us_median'],
                  item['p62_import_us_median'],item['p62_including_barrier_us_median']))
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
