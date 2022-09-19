import os

import cv2
import numpy as np
import torch
import clip
from PIL import Image
import time
import scann

import matplotlib as plt
import zipfile
from download import *


def loadData(fname):
    data = []
    f = open(fname, 'r+', encoding='utf-8')
    count = 0
    while count < 200000:
        line = f.readline()
        line = line.rstrip('\n')
        data.append(line)
        count += 1
        if not line:
            break
    f.close()

    data_clean1 = []
    for i in data:
        if len(i) > 230:
            data_clean1.append(i)

    data_clean = []
    for i in data_clean1:
        if i[0] != '!':
            data_clean.append(i)
    return data_clean


def getText_index(url):
    text = []
    for i in url:
        tmp = i.split("-")[0]
        text.append(tmp[79:-1])
    return text


def getText(url):
    text = []
    # for i in url:
    tmp = url.split("-")[0]
    # text.append(tmp[79:-1])
    # text.append(tmp)
    return tmp


def url_2_image(url):
    image_bytes = requests.get(url, verify=False).content
    image_np1 = np.frombuffer(image_bytes, dtype=np.uint8)
    image_np2 = cv2.imdecode(image_np1, cv2.IMREAD_COLOR)
    return image_np2


def scann_text(embd_input_text, searcher):
    # here we need to obtain the input string and convert it through clip model,
    # let's say we have the embedding input string as " embd_input_text " with
    # size 1 x embed_dimension
    # start = time.time()
    neighbors_text, distances_text = searcher.search_batched(
        embd_input_text)  # neighbors is the tuple with size 1 x nearest neighbor
    # end = time.time()
    # print(end - start)
    text_output_url = []
    for i in neighbors_text:
        text_output_url.append(url[i])
    return text_output_url





if __name__ == '__main__':


    # clip features
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    text = []
    text_model = []
    for filename in os.listdir(r'./image/image/'):
        image = preprocess(Image.open("./image/image/" + filename)).unsqueeze(0).to(device)
        # text = clip.tokenize(text).to(device)
        text1 = getText(filename)
        text_model.append(' '.join(text1.replace('_', ' ').split()))  # using processed name to embed
        with torch.no_grad():
            image_features1 = model.encode_image(image)
            # text_features = model.encode_text(text)

        image_features1 = image_features1.cpu().numpy()  # list to array
        image_features = np.vstack((image_features, image_features1))
    image_features = image_features[1:, :]
    print(image_features.shape)
    print(image_features[4:4 + 1, :])

    np.save("./result/text.npy", text)
    np.save("./result/text_model.npy", text_model)

    # scann_img
    searcher = scann.scann_ops_pybind.builder(image_features, 10, "dot_product").tree(
        num_leaves=2000, num_leaves_to_search=100, training_sample_size=250000).score_ah(
        2, anisotropic_quantization_threshold=0.2).reorder(100).build()
    list_save = []
    for i in range(10000):
        # text_output_url = scann_text(embd_input_text, searcher)
        neighbors = scann_img(i, url, searcher, image_features)
        list_save.append(neighbors)

    np.save('./result/list_sav.npy', list_save)










