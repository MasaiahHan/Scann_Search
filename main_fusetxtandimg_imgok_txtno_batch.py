import os
import time
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


def scann_img(i, searcher, image_features):
    # image scann
    #     for i in range(len(url)):  # input_url is users input
    #         if input_url == url[i]:
    #             index = i
    #             break

    neighbors_img, distances_img = searcher.search_batched(image_features[i:i + 1, :])
    return neighbors_img[0]


if __name__ == '__main__':
    time1 = time.time()
    # clip features
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # image = preprocess(image.unsqueeze(0)).to(device)
    # image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  #  convert to pil

    text = []
    text_model = []
    url = []
    image = []
    batch_size = 64
    count = 0
    save_interval = 10000

    for filename in os.listdir(r'./image/image/'):
        count += 1
        image1 = preprocess(Image.open("./image/image/" + filename))  # tensor type
        image1 = image1.numpy()
        image.append(image1)
        # text = clip.tokenize(text).to(device)
        text1 = getText(filename)
        url.append("https://iartai.com/destfolder/webp/image/" + filename)
        text.append(text1)  # keep origin name to track
        text_model.append(' '.join(text1.replace('_', ' ').split()))  # using processed name to embed

        if (count % save_interval == 0):
            image_features = np.zeros((1, 512))
            if (len(image) % batch_size == 0):
                length = len(image) // batch_size
            else:
                length = (len(image) // batch_size) + 1
            for i in range(length):
                image2 = torch.tensor(np.array(image[i * batch_size:(i + 1) * batch_size - 1])).to(device)
                with torch.no_grad():
                    image_features1 = model.encode_image(image2)
                # text_features = model.encode_text(text)
                image_features2 = image_features1.cpu().numpy()  # list to array
                image_features = np.vstack((image_features, image_features2))
            image_features = image_features[1:, :]
            np.save("./result/image_features_batch_" + str(count // save_interval) + ".npy", image_features)
            np.save("./result/url_batch_" + str(count // save_interval) + ".npy", url)
            np.save("./result/text_batch_" + str(count // save_interval) + ".npy", text)
            np.save("./result/text_model_batch_" + str(count // save_interval) + ".npy", text_model)
            image = []
            text = []
            text_model = []
            url = []

    time2 = time.time()
    print(time2 - time1)

    #     print(text1[0])
    #     print(url[1])
    # print(image_features.shape)
    # print(image_features[4:4 + 1, :])
    #     np.save("./result/url.npy", url)
    #     np.save("./result/text.npy", text)
    #     np.save("./result/text_model.npy", text_model)

    # scann_img
    searcher = scann.scann_ops_pybind.builder(image_features, 10, "dot_product").tree(
        num_leaves=2000, num_leaves_to_search=100, training_sample_size=250000).score_ah(
        2, anisotropic_quantization_threshold=0.2).reorder(100).build()
    list_save = []
    for i in range(10000):
        # text_output_url = scann_text(embd_input_text, searcher)
        neighbors = scann_img(i, url, searcher, image_features)
        list_save.append(neighbors)

    # np.save('./result/list_sav.npy', list_save)









