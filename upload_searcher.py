import numpy as np
import scann
import clip
import torch
import time

def scann_text(input_text, searcher, model, device):
    # here we need to obtain the input string and convert it through clip model,
    # let's say we have the embedding input string as " embd_input_text " with
    # size 1 x embed_dimension
    # start = time.time()
    token_text = clip.tokenize(input_text).to(device)
    with torch.no_grad():
        embd_text1 = model.encode_text(token_text)
    embd_text = embd_text1.cpu().numpy()
    neighbors_text, distances_text = searcher.search_batched(
        embd_text)  # neighbors is the tuple with size 1 x nearest neighbor

    return neighbors_text[0]




#text_features = np.load("./result/text_features.npy")
#url = np.load("./result/url.npy")


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
searcher_txt = scann.scann_ops_pybind.load_searcher("./result/searcher_txt")


neighbors_txt = scann_text(
        input_text="a cat with a long tail",
        searcher=searcher_txt, model=model, device=device)
