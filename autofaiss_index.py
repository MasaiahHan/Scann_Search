import faiss
import numpy as np
from autofaiss import build_index,tune_index
import time


# embeddings = np.float32(np.random.rand(6000000, 512))

# t1=time.time()
# # Example on how to build a memory-mapped index and load it from disk

# _, index_infos = build_index(
#     embeddings,
#     save_on_disk=True,
#     should_be_memory_mappable=True,
#     index_path="./result/faiss/knn.index",
#     max_index_memory_usage="6G",
#     max_index_query_time_ms=10,
#     index_key="OPQ256_768,IVF16384_HNSW32,PQ256x8"
# )

# t2 = time.time()
# print(t2-t1)







# index = faiss.read_index("./result/faiss/knn.index", faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)

# #index, index_infos = build_index(embeddings, save_on_disk=False)

# query = np.float32(np.random.rand(1, 512))
# t1=time.time()
# _, I = index.search(query, 1)

# t2=time.time()
# print(I)
# print(t2-t1)




  # load a flat (CPU) index
res = faiss.StandardGpuResources()
index=tune_index(index_path="./result/faiss/knn.index",index_key ="Flat" ,output_index_path="./result/faiss/knn_tune.index")
#index = faiss.read_index("./result/faiss/knn.index", faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
#index = faiss.read_index("./result/faiss/knn.index")
# make it into a gpu index

gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index)
#faiss.set_index_parameter(gpu_index, "nprobe", nprob)
#gpi_index_flat.setNumProbes(5)

xq = np.random.random((1, 512)).astype('float32')
t1=time.time()
k = 4                          # we want to see 4 nearest neighbors
_, I = gpu_index_flat.search(xq, k)  # actual search
t2 = time.time()
print(I)
print(t2-t1)

