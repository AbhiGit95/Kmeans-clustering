from PIL import Image
import numpy as np
import time
import argparse


def Kmeans(centroids, pixm, fl):
    ncentroids = []
    ml = []
    for cent in centroids:
        pixnormal = pixm - np.array(cent)
        #pixnormal = np.apply_along_axis(np.linalg.norm, 2, pixnormal)
        pixnormal = np.linalg.norm(pixnormal, axis=2)
        ml.append(pixnormal)
    ml = np.array(ml)
    result = np.argmin(ml, axis=0)
    pixm_rgb = pixm.reshape(pixm.shape[0]*pixm.shape[1], pixm.shape[2])
    for i in range(int(ml.shape[0])):
        c = (result == i)
        d = c.reshape(pixm.shape[0]*pixm.shape[1], 1)
        e = np.repeat(d, 3, axis=1)
        e = np.invert(e)
        f = np.ma.array(pixm_rgb, mask=e)
        ncentroids.append(np.mean(f, axis=0).data)
        if fl == True:
            e = np.invert(e)
            pixm_rgb[:, 0][e[:, 0]] = int(centroids[i][0])
            pixm_rgb[:, 1][e[:, 1]] = int(centroids[i][1])
            pixm_rgb[:, 2][e[:, 2]] = int(centroids[i][2])

    if fl == True:
        pixm_c = pixm_rgb.reshape([pixm.shape[0], pixm.shape[1], pixm.shape[2]])
        return pixm_c
    else:
        return [list(x) for x in ncentroids]

def main():

    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-image', '--img', type=str)
    parser.add_argument('-K', '--k', type=str)

    arg = parser.parse_args()
    img = arg.img
    K = int(arg.k)
    #opening the image
    #img = 'Koala.jpg'
    #K = 20
    image = Image.open(img)
    #get no. of pixels in length and width of the image
    width, height = image.size
    image.load()
    #load image as a numpy array
    pixel_mat = np.asarray(image, dtype="int32")

    flag = False
    converged = False
    #list of previous centroids
    previous_centroids = []
    #choosing random centroids
    for i in range(K):
        previous_centroids.append([np.random.randint(256), np.random.randint(256), np.random.randint(256)])

    print("Randomly Initialized centroids : ", previous_centroids)

    #run the program for 500 iterations and break if the algorithm converges
    for i in range(500):
        if (i+1) % 50 == 0:
            print("{}th iteration".format(i+1))
        new_centroids = Kmeans(previous_centroids, pixel_mat, flag)
        if previous_centroids == new_centroids:
            converged = True
            break
        else:
            previous_centroids = new_centroids
    #return the clustering matrix and convert to image
    flag = True
    new_mat = Kmeans(previous_centroids, pixel_mat, flag)
    new_mat = new_mat.astype(np.uint8)
    image_new = Image.fromarray(new_mat, "RGB")
    image_new.save("Output_Image K = {}, E = {}.png".format(K, i + 1))
    if converged:
        print("Converging at {} iteration ".format(i + 1))
    else:
        print("Didnt converge but ran for {} iteration".format(i + 1))
    # end = time.time() - start
    # print("Time taken : {}".format(end))

main()