import time
from multiprocessing.pool import ThreadPool
import requests
import os
import PIL.Image as Image
from io import BytesIO
import cv2
import numpy as np

def download_image(url, our_dir):
    '''
    根据url下载图片
    :param url:
    :return: 返回保存的图片途径
    '''
    basename = os.path.basename(url)
    try:
        res = requests.get(url)
        if res.status_code == 200:
            print("download image successfully:{}".format(url))
            filename = os.path.join(our_dir, basename)
            #with open(filename+".png", "wb") as f:
            content = res.content

                # 使用Image解码为图片
                # image = Image.open(BytesIO(content))
                # image.show()
                # 使用opencv解码为图片
            content1 = np.asarray(bytearray(content), dtype="uint8")
            image = cv2.imdecode(content1, cv2.IMREAD_COLOR)
            #cv2.imwrite(filename +".jpeg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
            cv2.imwrite(filename[0:-4]+".webp" , image, [cv2.IMWRITE_WEBP_QUALITY, 85])
                # cv2.imshow("Image", image)
                # cv2.waitKey(1000)
                # f.write(content)
                # time.sleep(2)
                #f.write(content)
        return image
    except Exception as e:
        print(e)
        return None
    print("download image failed:{}".format(url))
    return None


def download_image_thread(url_list, our_dir, num_processes, remove_bad=False, Async=True):
    '''
    多线程下载图片
    :param url_list: image url list
    :param our_dir:  保存图片的路径
    :param num_processes: 开启线程个数
    :param remove_bad: 是否去除下载失败的数据
    :param Async:是否异步
    :return: 返回图片的存储地址列表
    '''
    # 开启多线程
    if not os.path.exists(our_dir):
        os.makedirs(our_dir)
    pool = ThreadPool(processes=num_processes)
    thread_list = []
    for image_url in url_list:
        if Async:
            out = pool.apply_async(func=download_image, args=(image_url, our_dir))  # 异步
        else:
            out = pool.apply(func=download_image, args=(image_url, our_dir))  # 同步
        thread_list.append(out)
    pool.close()
    pool.join()
    #获取输出结果
    image_list = []
    if Async:
        for p in thread_list:
            image = p.get()  # get会阻塞
            image_list.append(image)
    else:
        image_list = thread_list
    if remove_bad:
        image_list = [i for i in image_list if i is not None]
    return thread_list


def load_data_new(url):
    data = []
    f = open(url, 'r+', encoding='utf-8')

    a = 1
    while a < 200000000000000:
        line = f.readline()
        line = line.rstrip('\n')
        data.append(line)
        if not line:
            break

    f.close()

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


# def file_put_contents(fileName,content,modeName =''):
#
#     if(modeName == 'pappend'):
#       mode = "r+";
#     elif(modeName == 'append'):
#      mode = "a+";
#     elif(modeName == 'wb+'):
#      mode = "wb+";
#     else:
#      mode = "w+";
#
#     # fileObject = open(fileName,mode);
#     # fileObject.write(content);
#     # fileObject.close();
#     content1 = np.asarray(bytearray(content), dtype="uint8")
#     image = cv2.imdecode(content1, cv2.IMREAD_COLOR)
#     #cv2.imwrite(filename +".jpeg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
#     cv2.imwrite(fileName, image, [cv2.IMWRITE_WEBP_QUALITY, 85])
#
# def downImg(url):
#     saveDir = "./images/"
#     #url = url[0];
#     try:
#         requests.adapters.DEFAULT_RETRIES = 5
#         s = requests.session()
#         s.keep_alive = False
#         ir = s.get(url)
#         if ir.status_code == 200:
#             # wb+  保存二进制
#             fileName = os.path.basename(url)[:-4] +'.webp';
#             file_put_contents(saveDir+fileName,ir.content,"wb+");
#
#     except Exception as e:
#         print(e)
#
# data = load_data_new('./ImageLinksFinal.txt')
# pool = ThreadPool(4) # Sets the pool size to 4
# results = pool.map(downImg,data);
# pool.close();
# pool.join();