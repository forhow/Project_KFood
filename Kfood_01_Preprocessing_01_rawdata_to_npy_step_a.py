"""
    Preprocessing

        - Step. a : Making image Data files (.npy)

        ; Raw data(image files)를
        Data(X) : 64 * 64 * 3 array
        Target(Y) : 1 * 150 array (One-Hot Encoding) 로 변환하고

        변환된 array 를 모두 병합시켜 소분류별 Data file 을 생성
        (step b. 에서 모두 병합)


    1. Setting / Initialization

    2. Image file path listing

    3. Image resizing and Data Save
     a. Target(Y) Data One-hot encoding
     b. Image Data load / convert to RGB / resize
     c. X, Y data listing
     d. X, Y data reshape (check)
     e. Partial Data save to npy

"""


import glob
from PIL import Image
import numpy as np

'1. Setting / Initialization'

# Directory / Category setting
kfood_dir = './datasets/kfood/'
main_category_dir=['구이', '국', '기타', '김치', '나물', '떡', '만두', '면', '무침', '밥', '볶음', '쌈', '음청류', '장', 
                   '장아찌', '적', '전', '전골', '조림', '죽', '찌개', '찜', '탕', '튀김', '한과', '해물', '회']
sub_category_dir=[['갈비구이', '갈치구이', '고등어구이', '곱창구이', '닭갈비', '더덕구이', '떡갈비', '불고기', '삼겹살', '장어구이', '조개구이', '조기구이', '황태구이', '훈제오리'],
                  ['계란국', '떡국_만두국', '무국', '미역국', '북엇국', '시래기국', '육개장', '콩나물국'], 
                  ['과메기', '양념치킨', '젓갈', '콩자반', '편육', '피자', '후라이드치킨'], 
                  ['갓김치', '깍두기', '나박김치', '무생채', '배추김치', '백김치', '부추김치', '열무김치', '오이소박이', '총각김치', '파김치'],
                  ['가지볶음', '고사리나물', '미역줄기볶음', '숙주나물', '시금치나물', '애호박볶음'], 
                  ['경단', '꿀떡', '송편'], 
                  ['만두'],
                  ['라면', '막국수', '물냉면', '비빔냉면', '수제비', '열무국수', '잔치국수', '짜장면', '짬뽕', '쫄면', '칼국수', '콩국수'],
                  ['꽈리고추무침', '도라지무침', '도토리묵', '잡채', '콩나물무침', '홍어무침', '회무침'],
                  ['김밥', '김치볶음밥', '누룽지', '비빔밥', '새우볶음밥', '알밥', '유부초밥', '잡곡밥', '주먹밥'],
                  ['감자채볶음', '건새우볶음', '고추장진미채볶음', '두부김치', '떡볶이', '라볶이', '멸치볶음', '소세지볶음', '어묵볶음', '오징어채볶음', '제육볶음', '주꾸미볶음'],
                  ['보쌈'],
                  ['수정과', '식혜'], 
                  ['간장게장', '양념게장'],
                  ['깻잎장아찌'],
                  ['떡꼬치'],
                  ['감자전', '계란말이', '계란후라이', '김치전', '동그랑땡', '생선전', '파전', '호박전'],
                  ['곱창전골'], 
                  ['갈치조림', '감자조림', '고등어조림', '꽁치조림', '두부조림', '땅콩조림', '메추리알장조림', '연근조림', '우엉조림', '장조림', '코다리조림'],
                  ['전복죽', '호박죽'], 
                  ['김치찌개', '닭계장', '동태찌개', '된장찌개', '순두부찌개'], 
                  ['갈비찜', '계란찜', '김치찜', '꼬막찜', '닭볶음탕', '수육', '순대', '족발', '찜닭', '해물찜'],
                  ['갈비탕', '감자탕', '곰탕_설렁탕', '매운탕', '삼계탕', '추어탕'],
                  ['고추튀김', '새우튀김', '오징어튀김'], 
                  ['약과', '약식', '한과'], 
                  ['멍게', '산낙지'],
                  ['물회', '육회']]

# image size setting
image_w = 64
image_h = 64

# vstack 용도 zero array 생성  - Deprecated
# np_x = np.zeros(shape=(265, 265, 3), dtype=np.int8)
# np_y = np.zeros(shape=(150,), dtype=np.int8)

# X, Y array 데이터 저장
X = []
Y = []

# Raw data 디렉토리 경로 저장
dir_list = []


'2. Image file path listing'
for idex, categorie in enumerate(main_category_dir):
    for category in range(len(sub_category_dir)):
        for food in range(len(sub_category_dir[category])):
            if idex == category:
                image_dir = kfood_dir + categorie + '/' + sub_category_dir[category][food] + '/'
                dir_list.append(image_dir)


'3. Image resizing and Data Save'
count = 0
for idx, path in enumerate(dir_list):

    '3.a. Target Data One-hot encoding'
    label = [0 for i in range(150)]
    label[idx] = 1

    # file path listing
    files = glob.glob(path+"/*.jpg")

    for i, f in enumerate(files):
        try:
            '3.b. image load / convert to RGB / resize'
            img = Image.open(f)
            img = img.convert("RGB")
            img = img.resize((image_w, image_h))

            # convert to array
            data = np.asarray(img)

            '3.c. X, Y data listing'
            X.append(data)
            Y.append(label)

            # progress check
            count += 1
            if count % 500 == 0:
                print(count)

        except Exception as e:
            print(str(e))

    '3.d. X, Y data reshape (check)'
    X = np.array(X).reshape((-1,64,64,3))
    Y = np.array(Y).reshape((-1, 150))

    '3.e. Partial Data save to npy'
    np.save(f'./datasets/kfood_npys/kfood_image_data_X_{idx}.npy', X)
    np.save(f'./datasets/kfood_npys/kfood_image_data_Y_{idx}.npy', Y)

    # list initialization
    X = []
    Y = []

    # for progress notice
    print(f'file {idx} saved and variables initialized')


