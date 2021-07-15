"""
    Preprocessing

        - Step. b : .npy files merge

        ; Preprocessing step a. 에서 생성된 .npy 파일들을

        하나의 .npy 파일로 병합 및 Train/Test set 으로 split 해서 Dataset 완성


    1. Partial Data(X) .npy file Merge

    2. Partial Target(Y) .npy file Merge

    3. Data(X), Target(Y) Merge and Train / Test Split

"""


import numpy as np
from sklearn.model_selection import train_test_split


'1. Partial Data(X) .npy file Merge'

'1.a. Partial Data(X) 1st Merge'
start_lst = [0, 21, 41, 61, 81, 101, 121, 141]
end_lst = [21, 41, 61, 81, 101, 121, 141, 150]

for num in range(8):
    start = start_lst[num]
    end = end_lst[num]

    for i in range(start, end):
        if i == start:
            X = np.load(f'./datasets/kfood_npys/kfood_image_data_X_{start}.npy', allow_pickle=True)
            np.save(f'./datasets/kfood_npys/kfood_image_data_X_part_{num}.npy', X)
            print(f'file {i} / Total Shape : ', X.shape)
        else:
            X = np.load(f'./datasets/kfood_npys/kfood_image_data_X_part_{num}.npy', allow_pickle=True)
            npx_x = np.load(f'datasets/kfood_npys/kfood_image_data_X_{i}.npy', allow_pickle=True)
            X = np.vstack((X, npx_x))
            np.save(f'./datasets/kfood_npys/kfood_image_data_X_part_{num}.npy', X)
            print(f'file {i} / Total Shape : ', X.shape)
    print(f'part {num} completed')


'1.b. Partial Data(X) Final Merge'
for j in range(8):
    if j == 0:
        X = np.load(f'./datasets/kfood_npys/kfood_image_data_X_part_{j}.npy', allow_pickle=True)
        print(f'X : {X.shape}')
        np.save('./datasets/kfood_npys/kfood_image_data_X.npy', X)
        print(f'part {j} / Total Shape : ', X.shape)
    else:
        X = np.load('./datasets/kfood_npys/kfood_image_data_X.npy', allow_pickle=True)
        npx_x = np.load(f'datasets/kfood_npys/kfood_image_data_X_part_{j}.npy', allow_pickle=True)
        print(f'X : {X.shape}, npx : {npx_x.shape}')
        X = np.vstack((X, npx_x))
        np.save('./datasets/kfood_npys/kfood_image_data_X.npy', X)
        print(f'part {j} / Total Shape : ', X.shape)   # 완료 part 7 / Total Shape :  (150087, 64, 64, 3)



'2. Partial Target(Y) .npy file Merge'
npy_0 = np.load('datasets/kfood_npys/kfood_image_data_Y_0.npy', allow_pickle=True)
npy_1 = np.load('datasets/kfood_npys/kfood_image_data_Y_1.npy', allow_pickle=True)
Y = np.vstack((npy_0, npy_1))

for i in range(2, 150):
    npy_y = np.load(f'datasets/kfood_npys/kfood_image_data_Y_{i}.npy', allow_pickle=True)
    Y = np.vstack((Y, npy_y))
    print(Y.shape)

np.save('./datasets/kfood_npys/kfood_image_data_Y.npy', Y)



'3. Data(X), Target(Y) Merge and Train / Test Split'
X = np.load('./datasets/kfood_npys/kfood_image_data_X.npy', allow_pickle=True)
Y = np.load('./datasets/kfood_npys/kfood_image_data_Y.npy', allow_pickle=True)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
xy = (X_train, X_test, Y_train, Y_test)

np.save("./datasets/kfood_image_data_64_color.npy", xy)
