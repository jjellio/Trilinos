#!/usr/bin/env python3
"""Host-only synthetic result parser test, not an NVSHMEM performance test."""
import csv
from pathlib import Path
import subprocess
import sys
import tempfile

SCRIPT=Path(__file__).with_name('analyze_p71_profiles.py')
CASE = sys.argv[1] if len(sys.argv) > 1 else 'matrix'
assert CASE in ('matrix', 'topology')

def write_csv(path, names, records):
    with path.open('w', newline='') as f:
        writer=csv.DictWriter(f, names)
        writer.writeheader()
        writer.writerows(records)

with tempfile.TemporaryDirectory(prefix='p71-analyzer-test-') as tmp:
    root=Path(tmp)
    prefix=root/'p71_inter2_s100_pass1_ibrc_warp_k512'
    log=prefix.with_suffix('.log')
    rows=[]
    for section,method,base in [
        ('DIRECT','tpetra_doImport',60),
        ('DIRECT','descriptor_two_stream',70),
        ('P71_FULL','tpetra_barrier_plus_import',80),
        ('P71_FULL','p62_barrier_plus_import',95),
        ('P71_PROFILE','producer_cuda_kernel_event',45),
        ('P71_PROFILE','receiver_cuda_kernel_event',55),
    ]:
        for iteration in range(2):
            for rank in range(2):
                rows.append(dict(case=CASE, section=section, method=method,
                                 iteration=iteration, rank=rank,
                                 us=(base if rank==1 else base-2)))
    write_csv(Path(str(prefix)+f'_{CASE}_p62_samples.csv'),
              ['case','section','method','iteration','rank','us'], rows)
    lines=[]
    for section,method,value in [
        ('DIRECT','tpetra_doImport',60),('DIRECT','descriptor_two_stream',70),
        ('P71_FULL','tpetra_barrier_plus_import',80),
        ('P71_FULL','p62_barrier_plus_import',95),
        ('P71_PROFILE','producer_cuda_kernel_event',45),
        ('P71_PROFILE','receiver_cuda_kernel_event',55),
        ('P71_PROFILE','cudaDeviceSynchronize_host_wait',15),
        ('P71_PROFILE','instrumented_gpu_import_host',100),
        ('P71_PROFILE','post_import_probe_readback',2),
    ]:
        lines.append(f'[NVSHMEM P6.2 {section}] case={CASE} method={method} median={value:.3f} p10=1 p90=2 min=1 max=2')
    lines+=[f'[NVSHMEM P7.1 PROFILE] case={CASE} protocol=unchanged_put_signal_per_descriptor status=PASS',
            '[NVSHMEM P6.2] SUMMARY status=PASS',
            f'[NVSHMEM P6.2 SETUP] case={CASE}']
    log.write_text('\n'.join(lines)+'\n')
    probe_fields=['case','rank','epoch','kind','descriptor_id','peer_rank','peer_pe','route',
                  'doubles','first_us','second_us','poll_attempts','clock_khz','sample_stride']
    for rank in range(2):
        probes=[]
        for kind in ['pack_and_put_api_return','signal_poll_and_scatter','local_copy']:
            probes.append(dict(case=CASE,rank=rank,epoch=7,kind=kind,descriptor_id=0,
                               peer_rank=1-rank,peer_pe=1-rank,route='inter' if kind!='local_copy' else 'local',
                               doubles=512 if kind!='local_copy' else 1,
                               first_us=1.5,second_us=2.5,
                               poll_attempts=5 if kind=='signal_poll_and_scatter' else 0,
                               clock_khz=1500000,sample_stride=1))
        write_csv(Path(f'{prefix}_{CASE}_p71_rank{rank}_probes.csv'),probe_fields,probes)
    # Retain the TSV distinction that the real runner uses.
    manifest=root/'manifest.tsv'
    import csv as csv_module
    with manifest.open('w',newline='') as f:
        writer=csv_module.DictWriter(f,['run_id','pass','variant','chunk','transport','topology','status','log','output_prefix'],delimiter='\t')
        writer.writeheader();writer.writerow({'run_id':'0001','pass':1,'variant':'warp','chunk':512,'transport':'ibrc','topology':'inter2','status':0,'log':str(log),'output_prefix':str(prefix)})
    (root/'config.txt').write_text('np=2\n')
    run=subprocess.run([sys.executable,str(SCRIPT),str(root)],text=True,capture_output=True)
    assert run.returncode==0,(run.stdout,run.stderr)
    with (root/'runs.csv').open(newline='') as f:
        data=list(csv.DictReader(f))
    assert len(data)==1
    assert float(data[0]['mpi_import_us'])==60
    assert float(data[0]['p62_including_start_barrier_us'])==95
    assert int(data[0]['sampled_gpu_work_units'])==6
    with (root/'probe_distributions.csv').open(newline='') as f:
        probes=list(csv.DictReader(f))
    assert len(probes)==3
    # Deleting a rank-local probe must be an error, not silently passing.
    Path(f'{prefix}_{CASE}_p71_rank1_probes.csv').unlink()
    rejected=subprocess.run([sys.executable,str(SCRIPT),str(root)],text=True,capture_output=True)
    assert rejected.returncode!=0
print('[NVSHMEM P7.1 ANALYZER] RESULT=PASS')
