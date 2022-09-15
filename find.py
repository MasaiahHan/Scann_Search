import numpy as np




def find(index):
    url = np.load("./result/url.npy")
    list_save =np.load('./result/list_sav.npy')
    target_index = list_save[index]
    output = []
    for j in range(len(target_index)):
        output.append((target_index[j],url[target_index[j]]))
    return output


def randomsamp(num):
    url = np.load("./result/url.npy")
    randindex = np.random.randint(0,len(url),num)
    result = []
    for i in range(len(randindex)):
        tup = (randindex[i],url[i])
        result.append(tup)
    return result


#print(randomsamp(4))
print(find(2))