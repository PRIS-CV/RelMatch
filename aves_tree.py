import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn

trees = [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 1, 1], [3, 3, 2, 1], [4, 4, 3, 0], [5, 5, 4, 0], [6, 6, 5, 2], [7, 7, 6, 1], [8, 8, 7, 3], [9, 9, 8, 4], [10, 8, 7, 3], [11, 10, 9, 5], [12, 11, 10, 6], [13, 12, 1, 1], [14, 10, 9, 5], [15, 13, 6, 1], [16, 14, 11, 0], [17, 14, 11, 0], [18, 15, 12, 7], [19, 16, 11, 0], [20, 17, 13, 1], [21, 18, 12, 7], [22, 19, 11, 0], [23, 20, 14, 0], [24, 21, 14, 0], [25, 22, 15, 1], [26, 23, 10, 6], [27, 21, 14, 0], [28, 24, 16, 8], [29, 25, 14, 0], [30, 26, 2, 1], [31, 27, 13, 1], [32, 28, 17, 9], [33, 29, 18, 1], [34, 30, 17, 9], [35, 31, 17, 9], [36, 32, 19, 1], [37, 33, 17, 9], [38, 34, 20, 10], [39, 35, 20, 10], [40, 36, 21, 11], [41, 37, 22, 1], [42, 38, 20, 10], [43, 39, 23, 1], [44, 40, 24, 1], [45, 31, 17, 9], [46, 41, 25, 1], [47, 42, 26, 12], [48, 43, 17, 9], [49, 44, 27, 3], [50, 45, 28, 13], [51, 46, 20, 10], [52, 47, 29, 1], [53, 48, 17, 9], [54, 48, 17, 9], [55, 49, 17, 9], [56, 50, 20, 10], [57, 51, 30, 11], [58, 34, 20, 10], [59, 52, 31, 1], [60, 53, 31, 1], [61, 54, 32, 14], [62, 55, 33, 1], [63, 56, 1, 1], [64, 57, 34, 15], [65, 26, 2, 1], [66, 58, 24, 1], [67, 59, 0, 0], [68, 48, 17, 9], [69, 60, 25, 1], [70, 61, 14, 0], [71, 47, 29, 1], [72, 62, 35, 0], [73, 47, 29, 1], [74, 34, 20, 10], [75, 63, 25, 1], [76, 64, 36, 1], [77, 65, 33, 1], [78, 66, 25, 1], [79, 67, 33, 1], [80, 68, 37, 1], [81, 69, 16, 8], [82, 70, 38, 1], [83, 71, 2, 1], [84, 72, 16, 8], [85, 73, 39, 1], [86, 74, 31, 1], [87, 53, 31, 1], [88, 73, 39, 1], [89, 75, 40, 1], [90, 76, 16, 8], [91, 77, 40, 1], [92, 78, 6, 1], [93, 79, 40, 1], [94, 80, 41, 1], [95, 81, 42, 1], [96, 82, 29, 1], [97, 83, 16, 8], [98, 84, 43, 1], [99, 85, 33, 1], [100, 86, 25, 1], [101, 87, 2, 1], [102, 88, 44, 1], [103, 89, 45, 1], [104, 90, 25, 1], [105, 91, 46, 1], [106, 92, 46, 1], [107, 47, 29, 1], [108, 93, 18, 1], [109, 47, 29, 1], [110, 94, 33, 1], [111, 47, 29, 1], [112, 95, 25, 1], [113, 52, 31, 1], [114, 96, 47, 1], [115, 97, 27, 3], [116, 98, 48, 11], [117, 0, 0, 0], [118, 99, 25, 1], [119, 100, 12, 7], [120, 101, 14, 0], [121, 102, 14, 0], [122, 14, 11, 0], [123, 103, 49, 1], [124, 104, 27, 3], [125, 105, 50, 16], [126, 106, 42, 1], [127, 107, 34, 15], [128, 68, 37, 1], [129, 108, 20, 10], [130, 0, 0, 0], [131, 37, 22, 1], [132, 94, 33, 1], [133, 21, 14, 0], [134, 109, 12, 7], [135, 110, 31, 1], [136, 98, 48, 11], [137, 111, 0, 0], [138, 112, 50, 16], [139, 113, 51, 1], [140, 42, 26, 12], [141, 114, 45, 1], [142, 115, 52, 17], [143, 116, 53, 1], [144, 117, 18, 1], [145, 79, 40, 1], [146, 118, 54, 1], [147, 118, 54, 1], [148, 119, 54, 1], [149, 14, 11, 0], [150, 120, 11, 0], [151, 121, 33, 1], [152, 122, 20, 10], [153, 123, 49, 1], [154, 124, 16, 8], [155, 34, 20, 10], [156, 125, 1, 1], [157, 33, 17, 9], [158, 126, 17, 9], [159, 127, 15, 1], [160, 128, 20, 10], [161, 123, 49, 1], [162, 129, 24, 1], [163, 130, 55, 18], [164, 59, 0, 0], [165, 131, 56, 1], [166, 132, 57, 1], [167, 64, 36, 1], [168, 133, 58, 0], [169, 0, 0, 0], [170, 74, 31, 1], [171, 55, 33, 1], [172, 32, 19, 1], [173, 134, 59, 19], [174, 76, 16, 8], [175, 135, 20, 10], [176, 136, 27, 3], [177, 97, 27, 3], [178, 137, 60, 0], [179, 138, 57, 1], [180, 139, 61, 1], [181, 140, 33, 1], [182, 141, 62, 19], [183, 142, 63, 1], [184, 143, 34, 15], [185, 2, 1, 1], [186, 144, 64, 14], [187, 145, 52, 17], [188, 146, 65, 20], [189, 147, 8, 4], [190, 70, 38, 1], [191, 24, 16, 8], [192, 148, 36, 1], [193, 79, 40, 1], [194, 149, 66, 1], [195, 76, 16, 8], [196, 150, 8, 4], [197, 151, 67, 1], [198, 152, 68, 21], [199, 153, 69, 22]]

def get_tree_target(pair_1,pair_2):

    tree_target_list = []


    for i in range(pair_1.size(0)):

        if trees[pair_1[i]][0] == trees[pair_2[i]][0]:
            tree_target_list.append(0)

        elif trees[pair_1[i]][1] == trees[pair_2[i]][1]:
            tree_target_list.append(1)

        elif trees[pair_1[i]][2] == trees[pair_2[i]][2]:
            tree_target_list.append(2)
            
        elif trees[pair_1[i]][3] == trees[pair_2[i]][3]:
            tree_target_list.append(3)

        else:
            tree_target_list.append(4)



    tree_target_list = Variable(torch.from_numpy(np.array(tree_target_list)).cuda())
    

    return tree_target_list

