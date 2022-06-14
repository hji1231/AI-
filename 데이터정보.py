import pandas as pd

data = pd.read_csv('../archive/heart.csv') # csv 읽어오기

print(data.shape) # 데이터갯수와 속성갯수
print(data.info()) # 데이터 정보
print(data.nunique()) # 속성별 유니크 정보
print(data.describe()) # 속성별 값 정보
print(data.isnull().sum()) # 결측치 확인
print(data.head()) # 데이터 5개 확인
print(data['HeartDisease'].value_counts()) # 레이블 분포 확인

