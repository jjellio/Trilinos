#!/usr/bin/env python3
"""Validate matrix-free NVSHMEM node-scaling runs and produce vector SVG figures.

Standalone Python 3 stdlib: no matplotlib, numpy, pandas or network required.
Reads the raw all-rank/iteration Tpetra and P6.2 timing CSVs; checks actual
physical node, peer degree, off-node fractions, and three PASS markers.
Keeps P6.2 + its mandatory start MPI_Barrier separate from Import-only timing.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import math
from pathlib import Path
import re
import statistics
import sys
from xml.sax.saxutils import escape

SUCCESS_MARKERS = (
    '[NVSHMEM TOPOLOGY IMPORTER] status=PASS',
    '[NVSHMEM P6.2 CASE] case=topology status=PASS',
    '[NVSHMEM TOPOLOGY] SUMMARY status=PASS',
)


def read_csv(path, delimiter=','):
    with path.open(newline='') as file:
        return list(csv.DictReader(file, delimiter=delimiter))


def write_csv(path, rows, keys):
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def getfile(folder, saved_path):
    path = Path(saved_path)
    relocated = folder / path.name
    if relocated.is_file():
        return relocated
    if path.is_file():
        return path
    raise ValueError(f'missing run artifact: {path.name}')


def short_host(name):
    return name.strip().split('.')[0].lower()


def ensure_numeric(name, value, nonnegative=True):
    number = float(value)
    if not math.isfinite(number) or (nonnegative and number < 0):
        raise ValueError(f'invalid {name}={value}')
    return number


def median(values):
    return statistics.median(values)


def load_timings(path, ranks):
    # Three independent timing sections recorded on every participating PE.
    # Pair the P6.2 Import and its measured barrier by (iteration,rank) BEFORE
    # reducing to the maximum rank and then the iteration median.
    names = [('DIRECT', 'tpetra_doImport'),
             ('DIRECT', 'descriptor_two_stream'),
             ('BARRIER', 'outside_gpu_import_timing')]
    raw = {name: defaultdict(dict) for name in names}
    with path.open(newline='') as file:
        for row in csv.DictReader(file):
            if row['case'] != 'topology':
                raise ValueError(f'wrong timing case: {row["case"]}')
            key = (row['section'], row['method'])
            if key not in raw:
                continue
            rank = int(row['rank'])
            epoch = int(row['iteration'])
            if not 0 <= rank < ranks or epoch < 0:
                raise ValueError('invalid rank/epoch in timing samples')
            if rank in raw[key][epoch]:
                raise ValueError('duplicate rank/epoch timing')
            raw[key][epoch][rank] = ensure_numeric('time in µs', row['us'])
    ref_sets = [set(raw[key]) for key in names]
    if not ref_sets[0] or any(x != ref_sets[0] for x in ref_sets):
        raise ValueError('missing or unmatched epoch sets in timing data')
    for key, epochs in raw.items():
        for epoch, by_rank in epochs.items():
            if set(by_rank) != set(range(ranks)):
                raise ValueError(f'missing rank/epoch for {key} iteration {epoch}')
    per_epoch = []
    for epoch in sorted(ref_sets[0]):
        mpi = max(raw[names[0]][epoch].values())
        p62 = max(raw[names[1]][epoch].values())
        start_barrier = max(raw[names[2]][epoch].values())
        full = max(raw[names[1]][epoch][r] + raw[names[2]][epoch][r]
                   for r in range(ranks))
        per_epoch.append((mpi, p62, start_barrier, full))
    mpi = median([s[0] for s in per_epoch])
    p62 = median([s[1] for s in per_epoch])
    full = median([s[3] for s in per_epoch])
    if mpi <= 0 or p62 <= 0 or full <= 0:
        raise ValueError('zero or negative completed Import timing')
    return dict(mpi_us=mpi, p62_us=p62,
                barrier_max_us=median([s[2] for s in per_epoch]),
                p62_full_us=full, speedup_full=mpi/full,
                speedup_import_only=mpi/p62,
                epochs=len(per_epoch), raw_rank_timing_rows=3*ranks*len(per_epoch))


def validate_traffic(path, manifest):
    ranks = int(manifest['pe_count'])
    n_nodes, ppn = int(manifest['nodes']), int(manifest['ppn'])
    degree_off, degree_intra = int(manifest['off_degree']), int(manifest['intra_degree'])
    expected_remote = int(manifest['remote_bytes_per_pe'])
    expected_cross = int(manifest['cross_bytes_per_pe'])
    entries = read_csv(path)
    if len(entries) != ranks*ranks:
        raise ValueError(f'traffic matrix requires {ranks*ranks} directed cells, got {len(entries)}')
    seen = set()
    counts_in = [0]*ranks
    counts_out = [0]*ranks
    cross_in = [0]*ranks
    cross_out = [0]*ranks
    deg_off_in = [0]*ranks
    deg_off_out = [0]*ranks
    deg_intra_in = [0]*ranks
    deg_intra_out = [0]*ranks
    hosts = {}
    edge_bytes = defaultdict(list)
    for entry in entries:
        s, r = int(entry['sender_rank']), int(entry['receiver_rank'])
        if s < 0 or r < 0 or s >= ranks or r >= ranks or (s,r) in seen:
            raise ValueError('duplicate or invalid traffic pair')
        seen.add((s,r))
        size = int(entry['bytes'])
        if size < 0 or size != 8*int(entry['doubles']):
            raise ValueError('inconsistent traffic byte count')
        sender_host, receiver_host = short_host(entry['sender_node']), short_host(entry['receiver_node'])
        for rank, host in ((s,sender_host),(r,receiver_host)):
            if rank in hosts and hosts[rank] != host:
                raise ValueError('one MPI rank mapped to multiple physical nodes')
            hosts[rank] = host
        if s == r:
            if size: raise ValueError('unexpected self-traffic')
            continue
        off = sender_host != receiver_host
        if size:
            counts_out[s] += size
            counts_in[r] += size
            edge_bytes[s].append(size)
            if off:
                cross_out[s] += size
                cross_in[r] += size
                deg_off_out[s] += 1
                deg_off_in[r] += 1
            else:
                deg_intra_out[s] += 1
                deg_intra_in[r] += 1
    selected = set(short_host(x) for x in manifest['selected_hosts'].split(','))
    if set(hosts.values()) != selected or len(selected) != n_nodes:
        raise ValueError('MPI traffic hosts do not equal the selected physical node subset')
    placements = defaultdict(int)
    for host in hosts.values(): placements[host] += 1
    if any(placements[host] != ppn for host in selected):
        raise ValueError('wrong actual processes per physical node')
    for rank in range(ranks):
        if counts_out[rank] != expected_remote or counts_in[rank] != expected_remote:
            raise ValueError('wrong per-PE total remote byte volume')
        if cross_out[rank] != expected_cross or cross_in[rank] != expected_cross:
            raise ValueError('wrong per-PE cross-node byte volume')
        if (deg_off_in[rank] != degree_off or deg_off_out[rank] != degree_off or
            deg_intra_in[rank] != degree_intra or deg_intra_out[rank] != degree_intra):
            raise ValueError('actual partner degree differs from requested graph')
        if manifest['volume'] == 'per-peer':
            if any(x != int(manifest['bytes_setting']) for x in edge_bytes[rank]):
                raise ValueError('per-peer packet size mismatch')
        elif edge_bytes[rank] and max(edge_bytes[rank])-min(edge_bytes[rank]) > 8:
            raise ValueError('fixed-total peer sizes are not balanced within one double')
    return dict(matrix_cells=len(entries), physical_host_count=len(selected),
                checked_rank_count=len(hosts))


def validate_log(path, manifest):
    text = path.read_text(errors='replace')
    missing = [marker for marker in SUCCESS_MARKERS if marker not in text]
    if missing:
        raise ValueError('missing completion marker(s): ' + ', '.join(missing))
    m = re.search(r'^\[NVSHMEM TOPOLOGY\] nranks=(\d+) nodes=(\d+) ppn=(\d+) '
                  r'offDegree=(\d+) intraDegree=(\d+)\b[^\n]*', text, re.M)
    if not m:
        raise ValueError('missing topology metadata line')
    actual = list(map(int, m.groups()))
    expected = [int(manifest[k]) for k in ('pe_count','nodes','ppn','off_degree','intra_degree')]
    if actual != expected:
        raise ValueError(f'topology header differs: got {actual}, expected {expected}')
    return text


def summarize_run(folder, manifest):
    if manifest['status'] != '0':
        raise ValueError(f'bad/absent MPI exit status {manifest["status"]}')
    prefix = Path(manifest['output_prefix'])
    log = getfile(folder, manifest['log'])
    if log.name != prefix.name + '.log':
        raise ValueError('manifest log/output prefix mismatch')
    validate_log(log, manifest)
    matrix = validate_traffic(getfile(folder, str(prefix)+'_traffic.csv'), manifest)
    timings = load_timings(getfile(folder, str(prefix)+'_topology_p62_samples.csv'),
                           int(manifest['pe_count']))
    return {**manifest, **matrix, **timings}


def aggregate(rows):
    keys = ['nodes','ppn','scenario','off_degree','intra_degree','volume',
            'bytes_setting','remote_bytes_per_pe','cross_bytes_per_pe',
            'transport','source','target','variant','chunk']
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(row)
    metrics = ['mpi_us','p62_us','p62_full_us','barrier_max_us',
               'speedup_full','speedup_import_only']
    result = []
    for values, sample in sorted(grouped.items()):
        record = dict(zip(keys, values))
        record['launches'] = len(sample)
        record['epochs_per_launch_min'] = min(int(s['epochs']) for s in sample)
        for key in metrics:
            data = [float(s[key]) for s in sample]
            record[key+'_median'] = median(data)
            record[key+'_min'] = min(data)
            record[key+'_max'] = max(data)
        result.append(record)
    return result


# Publication-style SVG plotter, using only the Python standard library.
COLORS = {'off1':'#2164ab','off4':'#bb4a27','mixed4':'#7656a5',
          'intra2':'#738394','intra4':'#374852','intra4multi':'#71926f'}


def svg_text(x, y, text, *, size=14, anchor='start', weight='normal', color='#24354a', rotate=None):
    transform = f' transform="rotate({rotate} {x:.2f} {y:.2f})"' if rotate else ''
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-size="{size}"'
            f' font-weight="{weight}" fill="{color}"{transform}>{escape(str(text))}</text>')


def svg_circle(x, y, color, *, radius=5, hollow=False):
    fill = '#ffffff' if hollow else color
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius}" fill="{fill}" stroke="{color}" stroke-width="2"/>'


def svg_plot(path, *, title, subtitle, series, y_label, max_node, is_speedup=False, x_min=1,
             legend_note='', annotate_local=True):
    """series: [{label, points:[(nodes,value,low,high)], color, dashed,hollow}]."""
    width, height = 1020, 640
    left,right,top,bottom = 106,970,127,535
    usable_w,usable_h=right-left,bottom-top
    values = [v for s in series for _,v,low,high in s['points'] for v in (v,low,high)]
    if not values:
        return False
    ymax = max(values+[1.0])
    if is_speedup:
        step = 0.5 if ymax <= 5 else 1.0
        ymax = math.ceil((ymax*1.14)/step)*step
        ymin = 0.0
    else:
        ymax = math.ceil(ymax*1.13/25)*25
        ymin = 0.0
        step = 25 if ymax <= 250 else 50 if ymax <= 500 else 100
    if max_node <= x_min:
        max_node=x_min+1
    def xp(n): return left + (float(n)-x_min)/(max_node-x_min)*usable_w
    def yp(v): return bottom - (float(v)-ymin)/(ymax-ymin)*usable_h
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}"'
             f' width="{width}" height="{height}" role="img" aria-label="{escape(title)}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<style>text{font-family:Arial,Helvetica,sans-serif}</style>',
             svg_text(left,38,title,size=23,weight='bold'),
             svg_text(left,68,subtitle,size=14,color='#556779')]
    v=0.0
    while v <= ymax+1e-9:
        y=yp(v)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" stroke="#e7ebf0" stroke-width="1"/>')
        label=f'{v:.1f}' if is_speedup else f'{v:.0f}'
        parts.append(svg_text(left-14,y+5,label,anchor='end',size=13))
        v += step
    if is_speedup:
        y=yp(1.0)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" stroke="#566777" stroke-width="1.3" stroke-dasharray="5 5"/>')
        parts.append(svg_text(right-3,y-9,'MPI parity (1.0×)',anchor='end',size=12,color='#556779'))
    for n in range(x_min,max_node+1):
        parts.append(svg_text(xp(n),bottom+28,str(n),anchor='middle',size=13))
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#5c6a79" stroke-width="1.5"/>')
    parts.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#5c6a79" stroke-width="1.5"/>')
    parts.append(svg_text((left+right)/2,bottom+65,'Physical nodes',anchor='middle',size=16,weight='bold'))
    parts.append(svg_text(29,(top+bottom)/2,y_label,anchor='middle',size=15,weight='bold',rotate=-90))
    for s in series:
        pts=sorted(s['points'],key=lambda p:p[0])
        color=s['color'];dashed=s.get('dashed',False)
        if len(pts)>=2:
            pathspec=' '.join(('M' if i==0 else 'L')+f' {xp(p[0]):.2f} {yp(p[1]):.2f}' for i,p in enumerate(pts))
            dash_attr=' stroke-dasharray="8 5"' if dashed else ''
            parts.append(f'<path d="{pathspec}" stroke="{color}" stroke-width="2.8" fill="none"{dash_attr}/>')
        for node,value,low,high in pts:
            x,y=xp(node),yp(value)
            if high-low>0.001:
                a,b=yp(low),yp(high)
                parts.append(f'<line x1="{x:.2f}" y1="{a:.2f}" x2="{x:.2f}" y2="{b:.2f}" stroke="{color}" stroke-width="1.4"/>')
                for end in (a,b):
                    parts.append(f'<line x1="{x-4:.2f}" y1="{end:.2f}" x2="{x+4:.2f}" y2="{end:.2f}" stroke="{color}" stroke-width="1.4"/>')
            parts.append(svg_circle(x,y,color,radius=4.4,hollow=s.get('hollow',False)))
    # Line keys in two rows, avoiding overlap with subtitle and plot.
    pos_x=left
    leg_y=92
    for s in series:
        label=s['label']
        approx_width=38+7*len(label)+23
        if pos_x+approx_width>right:
            leg_y += 23;pos_x=left
        color=s['color']
        dash_attr=' stroke-dasharray="8 5"' if s.get('dashed',False) else ''
        parts.append(f'<line x1="{pos_x}" y1="{leg_y-5}" x2="{pos_x+27}" y2="{leg_y-5}" stroke="{color}" stroke-width="2.6"{dash_attr}/>')
        parts.append(svg_circle(pos_x+14,leg_y-5,color,radius=3.5,hollow=s.get('hollow',False)))
        parts.append(svg_text(pos_x+35,leg_y,label,size=12))
        pos_x+=approx_width
    if legend_note:
        parts.append(svg_text(left,height-18,legend_note,size=12,color='#556779'))
    parts.append('</svg>')
    path.write_text('\n'.join(parts)+'\n')
    return True


def make_figures(folder, agg):
    transports=sorted(set(r['transport'] for r in agg))
    created=[]
    for transport in transports:
        matches=[r for r in agg if r['transport']==transport]
        max_node=max(int(r['nodes']) for r in matches)
        volume_labels=set((r['volume'], r['bytes_setting']) for r in matches)
        if len(volume_labels)==1:
            volume_mode, bytes_setting=next(iter(volume_labels))
            volume_description = (f'{int(bytes_setting)/1024:g} KiB imported per PE' if volume_mode=='fixed-total'
                                  else f'{int(bytes_setting)/1024:g} KiB per communication partner')
        else:
            volume_description='Mixed traffic volumes (compare matching series only)'
        def select(scenario, metric):
            data=[r for r in matches if r['scenario']==scenario]
            return [(int(r['nodes']),float(r[metric+'_median']),
                     float(r[metric+'_min']),float(r[metric+'_max'])) for r in data]
        speed_series=[]
        for scenario,label in [('off1','PPN=1 · all off-node'),('off4','PPN=4 · all off-node')]:
            pts=select(scenario,'speedup_full')
            if pts:
                speed_series.append(dict(label=label,points=pts,color=COLORS[scenario]))
        for scenario,label in [('intra2','N=1 · 2 GPUs intra'),('intra4','N=1 · 4 GPUs intra')]:
            pts=select(scenario,'speedup_full')
            if pts:
                speed_series.append(dict(label=label,points=pts,color=COLORS[scenario],hollow=True))
        if svg_plot(folder/f'speedup_vs_nodes_{transport}.svg',
                    title='Tpetra MPI vs NVSHMEM: node scalability',
                    subtitle=f'{volume_description}; solid curves include required P6.2 barrier',
                    series=speed_series,y_label='Speedup over MPI (MPI / NVSHMEM)',max_node=max_node,
                    is_speedup=True,
                    legend_note='Independent MPI-launch medians; whiskers span observed launch range. One-node points are intra-node controls.'):
            created.append(f'speedup_vs_nodes_{transport}.svg')
        for ppn,scenario in [(1,'off1'),(4,'off4')]:
            data=[r for r in matches if r['scenario']==scenario]
            if ppn==4:
                data += [r for r in matches if r['scenario']=='intra4']
            if not data:continue
            dseries=[]
            for field,label,color,dashed in [
                ('mpi_us','Tpetra/MPI Import','#355aa8',False),
                ('p62_full_us','NVSHMEM + required barrier','#c04e27',False),
                ('p62_us','NVSHMEM Import only (diagnostic)','#c04e27',True)]:
                # The n=1 intra4 comparison is an isolated control point;
                # it is not connected across the off-node regime.
                points=select(scenario,field)
                dseries.append(dict(label=label,points=points,color=color,dashed=dashed))
                if ppn==4:
                    pts=select('intra4',field)
                    if pts:
                        dseries.append(dict(label=label+' · intra N=1',points=pts,
                                            color=color,hollow=True))
            name=f'latency_vs_nodes_ppn{ppn}_{transport}.svg'
            if svg_plot(folder/name,
                title=f'Completed Import latency: {ppn} GPU{"s" if ppn!=1 else ""} per physical node',
                subtitle=f'{volume_description}; node count changes the number of off-node partners',
                series=dseries,y_label='Per-iteration max-rank latency (µs)',max_node=max_node,
                legend_note='Barrier-inclusive NVSHMEM is the operational comparison; Import-only timing is shown for diagnosis.'):
                created.append(name)
        control_series=[]
        for scenario,label in [('intra2','PPN=2 intra'),('intra4','PPN=4 intra, N=1'),
                                ('intra4multi','PPN=4 intra-only'),('mixed4','PPN=4 mixed'),
                                ('off4','PPN=4 off-node only')]:
            pts=select(scenario,'speedup_full')
            if pts:
                control_series.append(dict(label=label,points=pts,color=COLORS[scenario],
                                           hollow=scenario in ('intra2','intra4')))
        if svg_plot(folder/f'local_and_mixed_controls_{transport}.svg',
                    title='Intra-node and mixed-traffic controls',
                    subtitle=f'{volume_description}, varying intra-node versus off-node traffic',
                    series=control_series,y_label='Speedup over MPI, barrier included (×)',max_node=max_node,
                    is_speedup=True,
                    legend_note='Do not pool intra-node and off-node series: the proportion of network traffic differs.'):
            created.append(f'local_and_mixed_controls_{transport}.svg')
    return created


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--no-plots',action='store_true',help='Validate and summarize only')
    args=parser.parse_args()
    folder=args.directory.resolve()
    manifest=folder/'manifest.tsv'
    if not manifest.exists():
        parser.error('manifest.tsv not found')
    entries=read_csv(manifest,delimiter='\t')
    if not entries:
        parser.error('no sweep entries in manifest')
    # During resume, the final occurrence of a case_id is authoritative.
    by_case={}
    for row in entries: by_case[row['case_id']]=row
    rows=[];issues=[]
    for case_id,entry in by_case.items():
        try: rows.append(summarize_run(folder,entry))
        except (KeyError,ValueError,OSError,OverflowError,TypeError) as error:
            issues.append(f'{case_id}: {error}')
    run_fields=list(entries[0]) + ['matrix_cells','physical_host_count','checked_rank_count',
               'mpi_us','p62_us','barrier_max_us','p62_full_us','speedup_full',
               'speedup_import_only','epochs','raw_rank_timing_rows']
    write_csv(folder/'scaling_runs.csv',rows,run_fields)
    agg=aggregate(rows)
    group_fields=(['nodes','ppn','scenario','off_degree','intra_degree','volume',
            'bytes_setting','remote_bytes_per_pe','cross_bytes_per_pe',
            'transport','source','target','variant','chunk','launches','epochs_per_launch_min']+
            [m+suffix for m in ('mpi_us','p62_us','p62_full_us','barrier_max_us',
                               'speedup_full','speedup_import_only')
             for suffix in ('_median','_min','_max')])
    write_csv(folder/'scaling_summary.csv',agg,group_fields)
    plots=[] if args.no_plots else make_figures(folder,agg)
    report=['# Physical-node scalability: MPI vs NVSHMEM P6.2', '',
            f'Completed and validated independent MPI launches: **{len(rows)} / {len(by_case)}**.',
            f'Failures or incomplete data: **{len(issues)}**.', '',
            '### Definitions', '',
            '- MPI: stock Tpetra `doImport` completed elapsed time, without any extra global barrier.',
            '- NVSHMEM Import-only: current P6.2 benchmark after its prestart barrier; not a deployable algorithm cost.',
            '- **NVSHMEM full:** per-rank elapsed `(P6.2 Import + its necessary MPI_Barrier)`, then maximum rank per epoch, then epoch median.',
            '- Speedup: median across independent MPI runs of each run’s `(MPI median / NVSHMEM median)`.',
            '- Each run’s latency is median of its per-iteration maximum-rank elapsed times; independent MPI launches are the replicates.',
            '- Source/target ownership and GIDs are identical for MPI and NVSHMEM; no sparse matrix is built.',
            '- Pure off-node profiles keep byte volume per PE constant in fixed-total mode; varying peer degree changes per-peer payload size.',
            '- Intra-node-only and mixed traffic are separate comparison classes, not added to the pure off-node scalability curves.',
            '', '## Run-level median summary', '',
            '| Nodes | GPUs/node | Pattern | Cross-node partners | Intra partners | MPI (µs) | P6.2 Import (µs) | P6.2 + barrier (µs) | Realistic speedup | Launches |',
            '|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in agg:
        report.append('| {} | {} | {} ({}) | {} | {} | {:.2f} | {:.2f} | {:.2f} | {:.2f}× | {} |'.format(
           row['nodes'],row['ppn'],row['scenario'],row['transport'],row['off_degree'],
           row['intra_degree'],row['mpi_us_median'],row['p62_us_median'],
           row['p62_full_us_median'],row['speedup_full_median'],row['launches']))
    report.extend(['','## Generated figures','']+['- `'+name+'`' for name in plots]+['',
      '## Interpretation caution', '',
      'A P6.2 result excluding its MPI barrier does **not** establish a realistic speedup over Tpetra/MPI.',
      'The start-barrier-inclusive P6.2 measurement is an experimental algorithm cost, not yet a production-quality no-barrier Import.',
      'An Nsight Systems trace is required to establish actual GPU kernel/NIC overlap; raw timing records alone cannot do this.', ''])
    error_path=folder/'scaling_analysis_errors.txt'
    if issues:
        error_path.write_text('\n'.join(issues)+'\n')
        report+=['## Missing/invalid runs','']+['- '+x for x in issues]+['']
    elif error_path.exists():
        error_path.unlink()
    (folder/'scaling_report.md').write_text('\n'.join(report)+'\n')
    print(f'[NODE SCALING ANALYSIS] launches={len(by_case)} PASS={len(rows)} failed={len(issues)}')
    print(f'Validated {sum(int(row["raw_rank_timing_rows"]) for row in rows)} rank-timing records; '
          f'{len(agg)} configurations; created {len(plots)} SVG figures.')
    for row in agg:
        print(f'  nodes={row["nodes"]:>2} PPN={row["ppn"]} {row["scenario"]:<12} '
              f'transport={row["transport"]:<9} mpi={row["mpi_us_median"]:>8.2f}µs '
              f'p62={row["p62_us_median"]:>8.2f}µs '
              f'p62+barrier={row["p62_full_us_median"]:>8.2f}µs '
              f'speedup={row["speedup_full_median"]:.2f}x')
    for err in issues: print('ERROR',err,file=sys.stderr)
    return 1 if issues else 0

if __name__ == '__main__':
    sys.exit(main())
