

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


def scann_img(i, searcher, image_features):
    # image scann
    #     for i in range(len(url)):  # input_url is users input
    #         if input_url == url[i]:
    #             index = i
    #             break

    neighbors_img, distances_img = searcher.search_batched(image_features[i:i + 1, :])
    return neighbors_img[0]


if __name__ == '__main__':

    # clip features
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # image = preprocess(image.unsqueeze(0)).to(device)
    # image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  #  convert to pil
    image_features = np.zeros((1, 512))
    text_features = np.zeros((1, 512))
    text = []
    text_model = []
    url = []
    for filename in os.listdir(r'./image/image/'):
        image = preprocess(Image.open("./image/image/" + filename)).unsqueeze(0).to(device)
        # text = clip.tokenize(text).to(device)
        url.append("https://iartai.com/destfolder/webp/image/" + filename)

        text1 = getText(filename)
        text.append(text1)  # keep origin name to track
        text2 = (' '.join(text1.replace('_', ' ').split()))[41:-1].split(' -')[0]
        text_model.append(text2)  # using processed name to embed
        text3 = clip.tokenize(text2).to(device)
        with torch.no_grad():
            image_features1 = model.encode_image(image)
            text_features1 = model.encode_text(text3)

        image_features1 = image_features1.cpu().numpy()  # list to array
        text_features1 = text_features1.cpu().numpy()

        image_features = np.vstack((image_features, image_features1))
        text_features = np.vstack((text_features, text_features1))

    image_features = image_features[1:, :]
    text_features = text_features[1:, :]

    print(text_features.shape)
    print(text_features[4:4 + 1, :])
    np.save("./result/url.npy", url)
    np.save("./result/text.npy", text)
    np.save("./result/text_model.npy", text_model)
    np.save("./result/text_features.npy", text_features)

    # scann_img
    searcher = scann.scann_ops_pybind.builder(image_features, 10, "dot_product").tree(
        num_leaves=2000, num_leaves_to_search=100, training_sample_size=250000).score_ah(
        2, anisotropic_quantization_threshold=0.2).reorder(100).build()
    list_save = []
    for i in range(len(url)):
        # text_output_url = scann_text(embd_input_text, searcher)
        neighbors = scann_img(i, searcher, image_features)
        list_save.append(neighbors)

    np.save('./result/list_sav.npy', list_save)












