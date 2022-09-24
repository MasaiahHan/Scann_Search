
import json
import os
import numpy as np


def read_json(dir):
    return json.load(open(dir, 'r', encoding="utf-8"))

def convert_from_json(path,save_path):
    # read file name
    fileList=os.listdir(path)
    img_embedding = np.zeros((1,512))
    url = []
    txt_embedding = np.zeros((1,512))
    for filename in fileList:
        data = read_json(path+filename)
        img_embed = data['img_embed']
        text_embed = data['text_embed']
        urll = data['key']

        for dic in img_embed.values():
            img_embedding = np.vstack((img_embedding, dic))
        for dicc in text_embed.values():
            txt_embedding = np.vstack((txt_embedding,dicc))
        for diccc in urll.values():
            url.append(diccc)
    url = np.array(url)
    img_embedding = img_embedding[1:, :]
    txt_embedding = txt_embedding[1:, :]
    np.save(save_path+"img_embedding.npy", img_embedding)
    np.save(save_path + "txt_embedding.npy", txt_embedding)
    np.save(save_path + "url.npy", url)

convert_from_json(path=,save_path=)
