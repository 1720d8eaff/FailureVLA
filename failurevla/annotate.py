"""Export frame annotation sheets and import completed human annotations."""
import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict
from .data import files,load_episode
from .representation import FAILURES


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='mode',required=True)
    export=sub.add_parser('export');export.add_argument('--data',required=True);export.add_argument('--output',required=True)
    imp=sub.add_parser('import');imp.add_argument('--csv',required=True);imp.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    if a.mode=='export':
        with out.open('w',newline='',encoding='utf-8') as stream:
            w=csv.writer(stream);w.writerow(['episode','frame','timestamp_s','failure','progress','stage'])
            for path in files(a.data):
                ep=load_episode(path)
                for i,t in enumerate(ep['t']):w.writerow([path.name,i,t,'','',''])
    else:
        grouped=defaultdict(list)
        with open(a.csv,newline='',encoding='utf-8') as f:
            for r in csv.DictReader(f):
                if r['failure'] not in FAILURES:raise ValueError('all rows require a reviewed failure label')
                progress=float(r['progress']);stage=int(r['stage']);frame=int(r['frame'])
                if not 0<=progress<=1 or not 0<=stage<5:raise ValueError('invalid progress/stage')
                grouped[r['episode']].append((frame,{'failure':r['failure'],'progress':progress,'stage':stage}))
        records=[]
        for episode,rows in grouped.items():
            rows.sort()
            if [r[0] for r in rows]!=list(range(len(rows))):raise ValueError('frame gaps or duplicates')
            records.append({'episode':episode,'source':'human_annotation','frames':[r[1] for r in rows]})
        out.write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf-8')

if __name__=='__main__':main()
