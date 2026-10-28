#!/usr/bin/env python3
"""Local synthetic MPI CLI/CSV fixture for the node scaling driver test.
NOT an NVSHMEM implementation, NOT real performance data.
"""
import csv
import os
from pathlib import Path
import re
import sys

args=sys.argv[1:]
if '--version' in args:
    print('Open MPI synthetic fixture (NOT a real MPI installation)')
    raise SystemExit(0)
if '--hostfile' not in args or '-np' not in args:
    print('Missing hostfile or -np in mock', file=sys.stderr)
    raise SystemExit(2)
file=Path(args[args.index('--hostfile')+1])
np=int(args[args.index('-np')+1])
name_slots=[]
for line in file.read_text().splitlines():
    words=line.split()
    if not words:continue
    name_slots.append((words[0],int(next(x for x in words[1:] if x.startswith('slots=')).split('=')[1])))
hosts=[host for host,slots in name_slots for _ in range(slots)]
if np != len(hosts):
    raise SystemExit('Bad mock np and host count')
if args[-1]=='/bin/hostname':
    print('\n'.join(hosts))
    raise SystemExit(0)
opts={}
for arg in args:
    if arg.startswith('--') and '=' in arg:
        name,value=arg[2:].split('=',1)
        opts[name]=value
prefix=Path(opts['out'])
nodes=len(name_slots)
ppn=name_slots[0][1]
off=int(opts['offnode-degree'])
intra=int(opts['intranode-degree'])
degree=off+intra
volume=opts['volume']
values_per_peer=int(opts['values-per-peer'])
total=int(opts['remote-values-total'])
counts=([values_per_peer]*degree if volume=='per-peer' else
        [total//degree+(1 if k<total%degree else 0) for k in range(degree)])
traffic=[]
edgecounts=[[0]*np for _ in range(np)]
for layer in range(degree):
    is_off=layer<off
    relative=layer if is_off else layer-off
    node_delta=(relative//ppn+1) if is_off else 0
    slot_delta=(relative%ppn) if is_off else relative+1
    for sender in range(np):
        node=sender//ppn;slot=sender%ppn
        receiver=((node+node_delta)%nodes)*ppn+(slot+slot_delta)%ppn
        edgecounts[sender][receiver]=counts[layer]
for sender in range(np):
    for receiver in range(np):
        a=edgecounts[sender][receiver]
        left,right=hosts[sender],hosts[receiver]
        traffic.append(dict(sender_rank=sender,receiver_rank=receiver,
                            sender_node=left,receiver_node=right,
                            route='self' if sender==receiver else 'off-node' if left!=right else 'intra-node',
                            doubles=a,bytes=a*8))
with open(f'{prefix}_traffic.csv','w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(traffic[0]))
    writer.writeheader();writer.writerows(traffic)
with open(f'{prefix}_rank_topology.csv','w') as f:
    f.write('rank,node_name\n')
    for r,h in enumerate(hosts): f.write(f'{r},{h}\n')
expected_remote=sum(counts)
expected_cross=sum(counts[:off])
print(f'[NVSHMEM TOPOLOGY] nranks={np} nodes={nodes} ppn={ppn} offDegree={off} intraDegree={intra} source={opts["source-pattern"]} ownedPerRank={opts["owned"]} remoteDoublesPerRank={expected_remote} crossNodeDoublesPerRank={expected_cross} localValues=0 status=PLAN_PASS')
print(f'[NVSHMEM TOPOLOGY IMPORTER] status=PASS matchedRanks={np} matrixConstructed=NO independentImporter=YES')
print(f'[NVSHMEM P6.2 CASE] case=topology status=PASS verifiedOutputs={np*4}')
print(f'[NVSHMEM TOPOLOGY] SUMMARY status=PASS nodes={nodes} ppn={ppn}')
prefix_text=prefix.name
m=re.match(r'p(\d+)_n',prefix_text)
passid=int(m.group(1)) if m else 1
method_times={
    ('DIRECT','tpetra_doImport'):35+degree*9+nodes*1.7+intra*2.0+passid*0.3,
    ('DIRECT','descriptor_two_stream'):47+degree*1.1+nodes*1.3+passid*0.4,
    ('BARRIER','outside_gpu_import_timing'):9+nodes*2.7+np*0.12+passid*0.2,
}
with open(f'{prefix}_topology_p62_samples.csv','w',newline='') as f:
    w=csv.writer(f)
    w.writerow(['case','section','method','iteration','rank','us'])
    reps=int(opts['reps'])
    for (section,method),base in method_times.items():
        for epoch in range(reps):
            for rank in range(np):
                val=base + (rank%4)*0.2+(epoch%3)*0.08
                w.writerow(['topology',section,method,epoch,rank,f'{val:.4f}'])
