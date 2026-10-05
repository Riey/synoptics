import json,sys
d=json.load(open(sys.argv[1]));D=d['durationMs']
s=[x for x in d['samples'] if x['clipT'] is not None and x['luma'] is not None]
pre={round((x['clipT']+D)/100):x['luma'] for x in s if x['clipT']<0}
res=[]
for off in range(-10,11):
  pairs=[(x['luma'],pre[round(x['clipT']/100)+off]) for x in s if 0<=x['clipT']<D and round(x['clipT']/100)+off in pre]
  if len(pairs)>5: res.append((sum(abs(a-b) for a,b in pairs)/len(pairs),off*100,len(pairs)))
res.sort();print('loop alignment (mean |luma diff|, offset ms, n):',[(round(a,2),b,c) for a,b,c in res[:4]])
