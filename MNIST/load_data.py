import struct
from PIL import Image, ImageDraw, ImageDraw2

width = 0
height = 0


def drawImage(arr, weight, height):
    image = Image.new('RGB', (weight, height))
    draw = ImageDraw.Draw(image)
    rgb = ((0, 0, 0), (255, 255, 255))
    for y in range(weight):
        for x in range(height):
            draw.point(
                (x, y), fill=rgb[0 if arr[y * height + x] == 0 else 1])
    image.show()
    return image


def decode_idx1_ubyte(idx1_ubyte_file):
    """
    解析idx1文件的通用函数
    :param idx1_ubyte_file: idx1文件路径
    :return: tuple
    """
    with open(idx1_ubyte_file, 'rb') as f:
        buffers = f.read()
        head = struct.unpack_from('>II', buffers, 0)
        offset = struct.calcsize('>II')
        sum_lable = head[1]
    return struct.unpack_from('>' + str(sum_lable) + 'B', buffers, offset)


def decode_idx3_ubyte(idx3_ubyte_file):
    """
    解析idx3文件的通用函数
    :param idx3_ubyte_file: idx3文件路径
    :return: tuple
    """
    global width
    global height
    with open(idx3_ubyte_file, 'rb') as f:
        buffers = f.read()
        head = struct.unpack_from('>IIII', buffers, 0)
        offset = struct.calcsize('>IIII')
        imgNum = head[1]
        width = head[2]
        height = head[3]
        bitsString = '>' + str(width * height * imgNum) + 'B'
        imgs = struct.unpack_from(bitsString, buffers, offset)
        # drawImage(imgs[(width * height):(width * height) * 2], width, height)
    return imgs


train_imgs = decode_idx3_ubyte('data/train-images.idx3-ubyte')
train_lab = decode_idx1_ubyte('data/train-labels.idx1-ubyte')
test_imgs = decode_idx3_ubyte('data/t10k-images.idx3-ubyte')
test_lab = decode_idx1_ubyte('data/t10k-labels.idx1-ubyte')
