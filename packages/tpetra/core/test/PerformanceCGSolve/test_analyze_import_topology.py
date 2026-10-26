#!/usr/bin/env python3
"""Host-only fixture for P7.1b raw-rank/graph validator; no MPI/GPU needed."""
import csv
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
SCRIPT = HERE/'analyze_import_topology.py'


def write(path,fields,rows,sep=','):
    with path.open('w',newline='') as f:
        writer = csv.DictWriter(f,fields,delimiter=sep)
        writer.writeheader()
        writer.writerows(rows)


with tempfile.TemporaryDirectory(prefix='p71b-analyzer-test-') as d:
    root = Path(d)
    prefix = root/'demo'
    traffic = []
    for sender in range(2):
        for receiver in range(2):
            n = 0 if sender == receiver else 128
            traffic.append(dict(sender_rank=sender,receiver_rank=receiver,
                                sender_node='host'+str(sender),receiver_node='host'+str(receiver),
                                route='self' if sender==receiver else 'off-node',
                                doubles=n,bytes=8*n))
    traffic_path = Path(str(prefix)+'_traffic.csv')
    write(traffic_path,list(traffic[0]),traffic)
    samples = []
    for meth,section,base in [('tpetra_doImport','DIRECT',50),
                              ('descriptor_two_stream','DIRECT',70),
                              ('outside_gpu_import_timing','BARRIER',20)]:
        for epoch in range(3):
            for rank in range(2):
                samples.append(dict(case='topology',section=section,method=meth,
                                    iteration=epoch,rank=rank,us=base + rank))
    write(Path(str(prefix)+'_topology_p62_samples.csv'), list(samples[0]), samples)
    log = prefix.with_suffix('.log')
    log.write_text('[NVSHMEM TOPOLOGY] nranks=2 nodes=2 ppn=1 offDegree=1 intraDegree=0 '
                   'source=contiguous ownedPerRank=1024 remoteDoublesPerRank=128 '
                   'crossNodeDoublesPerRank=128 localValues=0 status=PLAN_PASS\n'
                   '[NVSHMEM P6.2] SUMMARY status=PASS\n'
                   '[NVSHMEM TOPOLOGY] SUMMARY status=PASS\n')
    entry=dict(run_id='0001',pass_id='1',topology='inter2',volume='per-peer',degree='1',
               intra_degree='0',payload_bytes='1024',source='contiguous',target='remote-only',
               variant='warp',chunk='512',transport='ibrc',mode='direct',status='0',
               log=str(log),output_prefix=str(prefix))
    entry['pass'] = entry.pop('pass_id')
    write(root/'manifest.tsv', list(entry),[entry],sep='\t')
    (root/'config.txt').write_text('np=2\nppn=1\n')
    def call():
        return subprocess.run([sys.executable,str(SCRIPT),str(root)],
                              text=True,capture_output=True)
    a=call()
    assert a.returncode==0,(a.stdout,a.stderr)
    data=list(csv.DictReader((root/'runs.csv').open()))
    assert len(data)==1
    assert float(data[0]['mpi_import_us'])==51
    assert float(data[0]['p62_import_us'])==71
    assert float(data[0]['p62_including_barrier_us'])==92
    assert int(data[0]['cross_node_doubles_global'])==256
    traffic[1]['doubles']=127
    write(traffic_path,list(traffic[0]),traffic)
    b=call()
    assert b.returncode!=0,b.stdout
    assert 'traffic size' in b.stderr or 'communication volume' in b.stderr
print('[P7.1b ANALYZER] RESULT=PASS')
