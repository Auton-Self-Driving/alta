import numpy as np

SEMANTIC_COLOR_MAP = {
    0	: ["Unlabeled", ( 0, 0, 0)],
    1	: ["Building",	( 70, 70, 70)],
    2	: ["Fence",	(190, 153, 153)],
    3	: ["Other",	(250, 170, 160)],
    4	: ["Pedestrian",	(220, 20, 60)],
    5	: ["Pole",	(153, 153, 153)],
    6	: ["Road line",	(157, 234, 50)],
    7	: ["Road",	(128, 64, 128)],
    8	: ["Sidewalk",	(244, 35, 232)],
    9	: ["Vegetation",	(107, 142, 35)],
    10	: ["Car",	( 0, 0, 142)],
    11	: ["Wall",	(102, 102, 156)],
    12	: ["Traffic sign",	(220, 220, 0)]
}

CLASS_REMAP = {
    0	: 0,
    1	: 0,
    2	: 0,
    3	: 0,
    4	: 1,
    5	: 0,
    6	: 2,
    7	: 3,
    8	: 0,
    9	: 0,
    10	: 4,
    11	: 0,
    12	: 0
}

REDUCED_SEMANTIC_COLOR_MAP = {
    0	: ["Everything Else", ( 0, 0, 0)],
    1	: ["Pedestrian",	(220, 20, 60)],
    2	: ["Road line",	(157, 234, 50)],
    3	: ["Road",	(128, 64, 128)],
    4	: ["Car",	( 0, 0, 142)]
}

def reduce_classes(semantic_image):
    h, w = np.shape(semantic_image)
    # assert(d == 1)
    semantic_reduced_image = np.zeros_like(semantic_image)

    for i in range(h):
        for j in range(w):
            orig_class = semantic_image[i, j]
            new_class = int(CLASS_REMAP[orig_class])
            semantic_reduced_image[i, j] = new_class

    return semantic_reduced_image

def convert_to_one_hot(labels, num_classes):
    labels = np.squeeze(labels)
    h, w = labels.shape
    flattened_labels = labels.reshape((h*w))
    one_hot = np.zeros((flattened_labels.shape[0], num_classes))
    one_hot[np.arange(flattened_labels.shape[0]), flattened_labels] = 1
    one_hot = one_hot.reshape((h, w, -1))
    
    return one_hot

def convert_from_one_hot(one_hot):
    return np.argmax(one_hot, axis=2)

def convert_to_rgb(semantic_image, reduced_classes=False):
    h, w = np.shape(semantic_image)
    semantic_rgb_image = np.zeros((h, w, 3))

    if reduced_classes:
        semantic_map = REDUCED_SEMANTIC_COLOR_MAP
    else:
        semantic_map = SEMANTIC_COLOR_MAP
    for i in range(h):
        for j in range(w):
            label = semantic_image[i, j]
            rgb_tuple = semantic_map[label][1]
            # print("rgb_tuple", rgb_tuple)
            semantic_rgb_image[i, j, 0] = rgb_tuple[0]
            semantic_rgb_image[i, j, 1] = rgb_tuple[1]
            semantic_rgb_image[i, j, 2] = rgb_tuple[2]
    
    return semantic_rgb_image

if __name__ == "__main__":
    image = np.array([[1, 12], [0, 6]])
    reduced_image = reduce_classes(image)
    print(image, reduced_image)
    one_hot = convert_to_one_hot(reduced_image, num_classes=5)
    re_image = convert_from_one_hot(one_hot)
    rgb_image = convert_to_rgb(re_image, reduced_classes=True)
    print(image, reduced_image, one_hot, re_image, rgb_image) 
