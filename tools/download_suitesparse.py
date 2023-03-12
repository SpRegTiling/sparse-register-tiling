from third_party.ssgetpy.ssgetpy.query import search
import os

os.makedirs("/datasets/suitesparse", exist_ok=True)
result = search(nzbounds=(100000,1000000000), isspd=False, limit=10000000000, dtype='real')
print(result)

print(len(result))
result.download(extract=True,destpath="/datasets/suitesparse")