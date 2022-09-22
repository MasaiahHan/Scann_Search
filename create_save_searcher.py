import numpy as np
import scann
import clip
import torch
import time


# text_features = np.load("./result/text_features.npy")
# #url = np.load("./result/url.npy")
#     #scann_txt
text_features = np.random.randint(0,255,size=(6000000,512))  ## need to load
searcher_txt = scann.scann_ops_pybind.builder(text_features, 100, "dot_product").tree(
        num_leaves=2400, num_leaves_to_search=50, training_sample_size=250000).score_ah(
        2, anisotropic_quantization_threshold=0.2).reorder(150).build()
searcher_txt.serialize("./result/searcher_txt")



