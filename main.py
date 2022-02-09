import cv2
import os
import frameextractor
import handshape_feature_extractor
import numpy as np
import scipy.spatial as sp

BASE = os.path.dirname(os.path.join(__file__))

Train_Path = os.path.join(BASE,"traindata")
Train_Frames_Path =os.path.join(BASE,'trainFrames')

Test_Path = os.path.join(BASE,'test')
Test_Frames_Path = os.path.join(BASE,'testFrames')
Train_Dictionary_image_vector_mapping = {}
count=0

# =============================================================================
# Get the penultimate layer for trainig data
# =============================================================================

for i in sorted(os.listdir(Train_Path)):
    if i.endswith(".mp4"):
        full_filename = os.path.join(Train_Path,i)
        frameextractor.frameExtractor(full_filename,Train_Frames_Path, count-1)
        count = count+1

instance = handshape_feature_extractor.HandShapeFeatureExtractor.get_instance()

for i in sorted(os.listdir(Train_Frames_Path)):
    if i.endswith(".png"):
        img = cv2.imread(os.path.join(Train_Frames_Path, i))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        vec1 = instance.extract_feature(img)
        vec1 = np.squeeze(vec1)
        # print(vec1)
        Train_Dictionary_image_vector_mapping[i] = vec1

print("Train_Dictionary_image_vector_mapping")
print(Train_Dictionary_image_vector_mapping)

# =============================================================================
# Get the penultimate layer for test data
# =============================================================================
count2=0
for i in sorted(os.listdir(Test_Path)):
    if i.endswith(".mp4"):
        frameextractor.frameExtractor(os.path.join(Test_Path,i),Test_Frames_Path,count2-1)
        count2 = count2+1
    else:
        continue

Test_Dictionary_image_vector_mapping={}

for k in sorted(os.listdir(Test_Frames_Path)):
    if k.endswith(".png"):
        print(Test_Frames_Path+"/"+k)
        # name = os.path.join(os.getcwd()+"/testFrames", k)
        img2 = cv2.imread(Test_Frames_Path+"/"+k)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        vec2 = instance.extract_feature(img2)
        vec2 = np.squeeze(vec2)
        # print(vec2)
        Test_Dictionary_image_vector_mapping[k] = vec2
print("Test_Dictionary_image_vector_mapping")
print(Test_Dictionary_image_vector_mapping)

# =============================================================================
# Recognize the gesture (use cosine similarity for comparing the vectors)
# =============================================================================

Output_Dictionary_trainImage_testImage_mapping = {}
for i in Test_Dictionary_image_vector_mapping:
    minimum = 99999
    for j in Train_Dictionary_image_vector_mapping:
        cosine_similarity = sp.distance.cosine(Test_Dictionary_image_vector_mapping[i], Train_Dictionary_image_vector_mapping[j])
        if cosine_similarity<minimum:
            minimum = cosine_similarity
            Output_Dictionary_trainImage_testImage_mapping[i] = j

print("Output_Dictionary_trainImage_testImage_mapping")
print(Output_Dictionary_trainImage_testImage_mapping)

Out = {}
counter = 0
for i in Train_Dictionary_image_vector_mapping:
    Out[i] = counter
    counter += 1

print("Out")
print(Out)


Customized_dictionary_for_51videos = {}
for i in sorted(os.listdir(Train_Path)):
    if "FanDown" in i:
        Customized_dictionary_for_51videos[0] = 10
        Customized_dictionary_for_51videos[1] = 10
        Customized_dictionary_for_51videos[2] = 10
    if "FanOff" in i:
        Customized_dictionary_for_51videos[3] = 12
        Customized_dictionary_for_51videos[4] = 12
        Customized_dictionary_for_51videos[5] = 12
    if "FanOn" in i:
        Customized_dictionary_for_51videos[6] = 11
        Customized_dictionary_for_51videos[7] = 11
        Customized_dictionary_for_51videos[8] = 11
    if "FanUp" in i:
        Customized_dictionary_for_51videos[9] = 13
        Customized_dictionary_for_51videos[10] = 13
        Customized_dictionary_for_51videos[11] = 13
    if "LightOff" in i:
        Customized_dictionary_for_51videos[12] = 14
        Customized_dictionary_for_51videos[13] = 14
        Customized_dictionary_for_51videos[14] = 14
    if "LightOn" in i:
        Customized_dictionary_for_51videos[15] = 15
        Customized_dictionary_for_51videos[16] = 15
        Customized_dictionary_for_51videos[17] = 15
    if "Num0" in i:
        Customized_dictionary_for_51videos[18] = 0
        Customized_dictionary_for_51videos[19] = 0
        Customized_dictionary_for_51videos[20] = 0
    if "Num1" in i:
        Customized_dictionary_for_51videos[21] = 1
        Customized_dictionary_for_51videos[22] = 1
        Customized_dictionary_for_51videos[23] = 1
    if "Num2" in i:
        Customized_dictionary_for_51videos[24] = 2
        Customized_dictionary_for_51videos[25] = 2
        Customized_dictionary_for_51videos[26] = 2
    if "Num3" in i:
        Customized_dictionary_for_51videos[27] = 3
        Customized_dictionary_for_51videos[28] = 3
        Customized_dictionary_for_51videos[29] = 3
    if "Num4" in i:
        Customized_dictionary_for_51videos[30] = 4
        Customized_dictionary_for_51videos[31] = 4
        Customized_dictionary_for_51videos[32] = 4
    if "Num5" in i:
        Customized_dictionary_for_51videos[33] = 5
        Customized_dictionary_for_51videos[34] = 5
        Customized_dictionary_for_51videos[35] = 5
    if "Num6" in i:
        Customized_dictionary_for_51videos[36] = 6
        Customized_dictionary_for_51videos[37] = 6
        Customized_dictionary_for_51videos[38] = 6
    if "Num7" in i:
        Customized_dictionary_for_51videos[39] = 7
        Customized_dictionary_for_51videos[40] = 7
        Customized_dictionary_for_51videos[41] = 7
    if "Num8" in i:
        Customized_dictionary_for_51videos[42] = 8
        Customized_dictionary_for_51videos[43] = 8
        Customized_dictionary_for_51videos[44] = 8
    if "Num9" in i:
        Customized_dictionary_for_51videos[45] = 9
        Customized_dictionary_for_51videos[46] = 9
        Customized_dictionary_for_51videos[47] = 9
    if "SetThermo" in i:
        Customized_dictionary_for_51videos[48] = 16
        Customized_dictionary_for_51videos[49] = 16
        Customized_dictionary_for_51videos[50] = 16


result= []

if (len(Train_Dictionary_image_vector_mapping.keys()) == 17):
    for i in Output_Dictionary_trainImage_testImage_mapping:
        result.append(Out[Output_Dictionary_trainImage_testImage_mapping[i]])
else:
    for i in Output_Dictionary_trainImage_testImage_mapping:
        key = Output_Dictionary_trainImage_testImage_mapping[i]
        res = Out[key]
        result.append(Customized_dictionary_for_51videos[res])

print(result)
np.savetxt("Results.csv", result, delimiter=',', fmt="%d")