#!/usr/bin/env python3
"""End-to-end host-side test of the Bash driver, analyzer and SVG figures."""
from pathlib import Path
import csv
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

root=Path(__file__).resolve().parents[1]
runner=root/'run_node_scaling.sh'
analyzer=root/'analyze_node_scaling.py'
mock=Path(__file__).with_name('mock_mpirun.py')
with tempfile.TemporaryDirectory(prefix='node-scaling-mock-') as temporary:
    path=Path(temporary)
    out=path/'simulation'
    env=os.environ.copy()
    env.update(SCALE_HOSTS='h01 h02 h03 h04',SCALE_NODES='1 2 3',
               SCALE_PASSES='1',SCALE_WARMUP='2',SCALE_REPS='5',
               SCALE_PLOT='1',SCALE_DRY_RUN='0',SCALE_BIND='/bin/true',
               SCALE_EXE='/bin/true',SCALE_MPI=str(mock),
               SCALE_OUT=str(out),SCALE_TIMEOUT='10',SCALE_ROTATE_HOSTS='1',
               SCALE_TRANSPORTS='ibrc',SCALE_MIXED_NODES='2 4',
               SCALE_INTRA_ONLY_NODES='2 4')
    proc=subprocess.run(['bash',str(runner)],env=env,text=True,capture_output=True)
    assert proc.returncode == 0, (proc.stdout[-5000:],proc.stderr[-5000:])
    with (out/'manifest.tsv').open(newline='') as f:
        entries=list(csv.DictReader(f,delimiter='\t'))
    assert len(entries)==8,len(entries) # n=1:2; n=2:4; n=3:2
    assert len({x['case_id'] for x in entries})==8
    assert all(x['status']=='0' for x in entries)
    with (out/'scaling_runs.csv').open(newline='') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==len(entries)
    assert all(int(x['epochs'])==5 for x in rows)
    assert all(int(x['matrix_cells'])==int(x['pe_count'])**2 for x in rows)
    assert all(float(x['p62_full_us'])>float(x['p62_us']) for x in rows)
    with (out/'scaling_summary.csv').open(newline='') as f:
        summaries=list(csv.DictReader(f))
    assert len(summaries)==8
    assert all(int(x['launches'])==1 for x in summaries)
    names=['speedup_vs_nodes_ibrc.svg','latency_vs_nodes_ppn1_ibrc.svg',
           'latency_vs_nodes_ppn4_ibrc.svg','local_and_mixed_controls_ibrc.svg']
    for name in names:
        img=out/name
        assert img.exists(), name
        assert ET.parse(img).getroot().tag.endswith('svg'),name
    # Test resume doesn't create duplicate entries or run already completed work.
    env['SCALE_RESUME']='1'
    second=subprocess.run(['bash',str(runner)],env=env,text=True,capture_output=True)
    assert second.returncode == 0,(second.stdout[-3000:],second.stderr[-3000:])
    assert 'skipped=8' in second.stdout,second.stdout[-700:]
    with (out/'manifest.tsv').open(newline='') as f:
        assert len(list(csv.DictReader(f,delimiter='\t')))==8
    # Missing real P6.2 CASE marker MUST fail, even if topology passed.
    victim=Path(entries[0]['log'])
    original=victim.read_text()
    victim.write_text(original.replace('[NVSHMEM P6.2 CASE] case=topology status=PASS',
                                       '[NVSHMEM P6.2] SUMMARY status=PASS'))
    fail=subprocess.run([sys.executable,str(analyzer),str(out),'--no-plots'],
                        text=True,capture_output=True)
    assert fail.returncode!=0 and 'missing completion marker' in fail.stderr,fail.stderr
    victim.write_text(original)
    good=subprocess.run([sys.executable,str(analyzer),str(out),'--no-plots'],
                        text=True,capture_output=True)
    assert good.returncode==0,(good.stdout,good.stderr)
    assert not (out/'scaling_analysis_errors.txt').exists()
    # Corrupt one traffic edge and require explicit failure.
    traffic=Path(entries[0]['output_prefix']+'_traffic.csv')
    with traffic.open(newline='') as f:
        data=list(csv.DictReader(f))
    changed=False
    for row in data:
        if int(row['bytes'])>0:
            row['bytes']=str(int(row['bytes'])+8)  # corrupt bytes but not doubles
            changed=True
            break
    assert changed
    with traffic.open('w',newline='') as f:
        writer=csv.DictWriter(f,data[0].keys())
        writer.writeheader();writer.writerows(data)
    fail=subprocess.run([sys.executable,str(analyzer),str(out),'--no-plots'],
                        text=True,capture_output=True)
    assert fail.returncode != 0
print('[NODE SCALING HOST TEST] RESULT=PASS')
