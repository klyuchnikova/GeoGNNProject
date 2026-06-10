from pathlib import Path
from datetime import datetime,timedelta,timezone
import random
random.seed(42)
out=Path("data/synthetic");out.mkdir(parents=True,exist_ok=True)
with (out/"checkins.tsv").open("w") as f:
    base=datetime(2010,1,1,tzinfo=timezone.utc)
    for u in range(8):
        rows=[]
        for i in range(32):
            poi=(i*3+u+(i//5))%15
            t=base+timedelta(hours=8*i+u)
            lat=30.20+0.005*(poi%5);lon=-97.80+0.005*(poi//5)
            rows.append((u,t.strftime("%Y-%m-%dT%H:%M:%SZ"),lat,lon,poi))
        # SNAP files are commonly reverse chronological per user; preprocessing is order-independent.
        for r in reversed(rows): f.write("\t".join(map(str,r))+"\n")
with (out/"friends.tsv").open("w") as f:
    for u in range(8): f.write(f"{u}\t{(u+1)%8}\n")
print(out)
