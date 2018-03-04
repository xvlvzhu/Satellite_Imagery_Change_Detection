import numpy as np
import cv2
import tifffile as tiff

weight,height,channel=(256,256,3)
origin_size = (4000,15106)
img_size=256

def read_labeled_image_list(image_list_file,data_dir):

    f = open(image_list_file, 'r')
    images = []
    masks = []
    for line in f:
        try:
            image, mask = line.strip("\n").split(' ')
        except ValueError: # Adhoc for test.
            image = mask = line.strip("\n")
        images.append(data_dir + image)
        masks.append(data_dir + mask)
    return images, masks

def read_images_from_disk(images, masks):

    train_x = np.zeros([len(images),256,256,3])
    train_y = np.zeros([len(images),256,256,1])
    for i in range(len(images)):
        train_img = cv2.imread(images[i],cv2.IMREAD_UNCHANGED)
        w,h,c = train_img.shape
        train_x[i,0:w,0:h,:] =train_img
    for i in range(len(masks)):
        train_label = np.resize(cv2.imread(masks[i],cv2.IMREAD_UNCHANGED),[-1,256,1])
        w,h,c = train_label.shape
        train_y[i,0:w,0:h,:] =train_label
    return train_x,train_y

def read_tiff_from_disk(filename):
    return tiff.imread(filename)

def read_images_combination_from_disk(images, masks):

    train_x = np.zeros([len(images),256,256,8])
    train_y = np.zeros([len(images),256,256,1])
    for i in range(len(images)):
        train_img = cv2.imread(images[i],cv2.IMREAD_UNCHANGED)
        w,h,c = train_img.shape
        train_x[i,0:w,0:h,0:4] =train_img
    for i in range(len(masks)):
        train_label = np.resize(cv2.imread(masks[i],cv2.IMREAD_UNCHANGED),[-1,256,1])
        w,h,c = train_label.shape
        train_y[i,0:w,0:h,:] =train_label
    return train_x,train_y

def cut_and_combine_tif(data1,data2,label,img_size):

    row_num = int(len(data1)/img_size) + 1
    column_num = int(len(data1[0])/img_size)
    img_num = row_num*column_num

    result_x = np.zeros([img_num,img_size,img_size,8])
    result_y = np.zeros([img_num,img_size,img_size,1])
    for i in range(row_num):
        for j in range(column_num):
            if i==row_num-1:
                result_x[i*column_num+j,0:len(data1)%img_size,0:img_size] = np.concatenate((data1[i*img_size:i*img_size+img_size, j*img_size:j*img_size+img_size, :4],data2[i*img_size:i*img_size+img_size, j*img_size:j*img_size+img_size, :4]),2)
                result_y[i*column_num+j,0:len(data1)%img_size,0:img_size] = label[i*img_size:i*img_size+img_size, j*img_size:j*img_size+img_size].reshape([len(data1)%img_size,img_size,1])
            else:
                result_x[i*column_num+j] = np.concatenate((data1[i*img_size:i*img_size+img_size, j*img_size:j*img_size+img_size, :4],data2[i*img_size:i*img_size+img_size, j*img_size:j*img_size+img_size, :4]),2)
                result_y[i*column_num+j] = label[i*img_size:i*img_size+img_size, j*img_size:j*img_size+img_size].reshape([img_size,img_size,1])
            # return 0
    return result_x,result_y

if __name__ == '__main__':
    img,lb = read_labeled_image_list("../data/train.txt","E:/tianchi/preliminary_1/")
    print(img[1111])
    print(lb[1111])
    # train_x, train_y = read_images_from_disk(img,lb)
    # np.save("train_x",train_x)
    # np.save("train_y",train_y)

    data2017 = read_tiff_from_disk("E:/tianchi/20171105_quarterfinals/quarterfinals_2015.tif").transpose([1, 2, 0])
    data2015 = read_tiff_from_disk("E:/tianchi/20171105_quarterfinals/quarterfinals_2017.tif").transpose([1, 2, 0])
    # origin_size = data2017.shape
    # print(origin_size)
    # print(len(data2015[0]))
    # print(len(data2015))
    label = read_tiff_from_disk("E:/tianchi/20171105_quarterfinals/out_11_8_h.tiff")
    cut_result_x,cut_result_y = cut_and_combine_tif(data2017,data2015,label,img_size)
    print(cut_result_x.shape)
    print(cut_result_y.shape)
    np.save("train_x_tif8", cut_result_x)
    np.save("train_y_tif8", cut_result_y)
    # print(np.shape(np.reshape(cv2.imread(lb[1],cv2.IMREAD_UNCHANGED),[256,256,1])))