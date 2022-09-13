
import numpy as np
import torch
import clip
from PIL import Image
import time
import scann



def loadData(fname):
    data = []
    f = open(fname, 'r+', encoding='utf-8')
    count = 0
    while count<200000:
        line = f.readline()
        line = line.rstrip('\n')
        data.append(line)
        count += 1
        if not line:
            break
    f.close()

    data_clean1 = []
    for i in data:
        if len(i) >230:
            data_clean1.append(i)

    data_clean = []
    for i in data_clean1:
        if i[0] != '!':
            data_clean.append(i)
    return data_clean


def getText(url):
    text = []
    for i in url:
        tmp = i.split("-")[0]
        text.append(tmp[79:-1])
    return text


if __name__ == '__main__':

    # split text and url:
    url = loadData(fname = '/Users/hantianyang/Downloads/ImageLinksFinal.txt')
    text = getText(url)

    all_in = np.array(url)

    # get image
    not finishing!!!!!!!!!!!!!!!!!!!
    need to download from url list

    # clip features
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
    text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

    image_features = np.array(image_features)       # list to array
    text_features = np.array(text_features)

    all_in = np.hstack((all_in, image_features))       # array concat
    all_in = np.hstack((all_in, text_features))



# use scann.scann_ops.build() to instead create a TensorFlow-compatible searcher

    #text scann
    searcher = scann.scann_ops_pybind.builder(text_features, 10, "dot_product").tree(
        num_leaves=2000, num_leaves_to_search=100, training_sample_size=250000).score_ah(
        2, anisotropic_quantization_threshold=0.2).reorder(100).build()

        # here we need to obtain the input string and convert it through clip model,
        # let's say we have the embedding input string as " embd_input_text " with
        # size 1 x embed_dimension
    start = time.time()
    neighbors, distances = searcher.search_batched(embd_input_text)  # neighbors is the tuple with size 1 x nearest neighbor
    end = time.time()
    print(end - start)

    text_output_url = []
    for i in neighbors:
        text_output_url.append(all_in[i][2])

    # image scann



