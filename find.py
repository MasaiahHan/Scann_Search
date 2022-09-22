import numpy as np
import clip
import torch
import scann



def find(index):
    '''
    :param index: int
    :return: a list with tuple(index, url)
    '''
    url = np.load("./result/url.npy")
    list_save =np.load('./result/list_sav.npy')
    text = np.load('./result/text.npy')
    target_index = list_save[index]
    output = []
    for j in range(len(target_index)):
        output.append((target_index[j],url[target_index[j]], text[target_index[j]]))
    return output


def randomsamp(num):
    '''
    :param num: int , number of pictures you want to sample in the page
    :return: a list with tuple(index, url)
    '''
    url = np.load("./result/url.npy")
    randindex = np.random.randint(0,len(url),num)
    result = []
    for i in range(len(randindex)):
        tup = (randindex[i],url[randindex[i]])
        result.append(tup)
    return result



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


def text_find(input_text):
    '''
    :param input_text:  a string users input
    :return:  a list with tuple(index, url)
    '''
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    text_features = np.load("./result/text_features.npy")
    url = np.load("./result/url.npy")
    # scann_txt
    searcher_txt = scann.scann_ops_pybind.builder(text_features, 10, "dot_product").tree(
        num_leaves=2000, num_leaves_to_search=100, training_sample_size=250000).score_ah(
        2, anisotropic_quantization_threshold=0.2).reorder(100).build()

    neighbors_txt = scann_text(
        input_text=input_text,
        searcher=searcher_txt, model=model, device=device)

    output = []
    for j in range(len(neighbors_txt)):
        output.append((neighbors_txt[j], url[neighbors_txt[j]]))
    return output





def scann_img(searcher, image_features):
    # image scann
    #     for i in range(len(url)):  # input_url is users input
    #         if input_url == url[i]:
    #             index = i
    #             break

    neighbors_img, distances_img = searcher.search_batched(image_features)
    return neighbors_img[0]

def load_image_find(input_pil_img):
    '''
    :param input_text:  a string users input
    :return:  a list with tuple(index, url)
    '''
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    image_features = np.load("./result/img_features.npy")
    url = np.load("./result/url.npy")

    image = preprocess(input_pil_img).unsqueeze(0).to(device)   # input image
    with torch.no_grad():
        image_features1 = model.encode_image(image)   ##  input image feature
    image_features1 = image_features1.cpu().numpy()  # list to array
    # scann_image
    searcher_image = scann.scann_ops_pybind.builder(image_features, 10, "dot_product").tree(
        num_leaves=2000, num_leaves_to_search=100, training_sample_size=250000).score_ah(
        2, anisotropic_quantization_threshold=0.2).reorder(100).build()

    neighbors_txt = scann_img(
        image_features=image_features1,
        searcher=searcher_image)

    output = []
    for j in range(len(neighbors_txt)):
        output.append((neighbors_txt[j], url[neighbors_txt[j]]))
    return output


