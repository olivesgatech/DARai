import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# XML 파일 경로
file_path = "/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/datasets/utkinect/depth/s01_e01/depthImg190.xml"

# XML 파일 파싱
tree = ET.parse(file_path)
root = tree.getroot()
tag = os.path.basename(file_path).replace('.xml', '')

# <depthImg190> 태그 찾기
depth_node = root.find(tag)

if depth_node is None:
    raise ValueError("depthImg190 태그를 찾을 수 없습니다. XML 구조를 확인하세요.")

# <width>와 <height> 값 가져오기
width = int(depth_node.find("width").text)
height = int(depth_node.find("height").text)

# <data> 태그 찾기
data_node = depth_node.find("data")

if data_node is None or not data_node.text:
    raise ValueError("data 태그가 없거나 비어 있습니다. XML 파일을 확인하세요.")

# 데이터 문자열 가져오기 및 변환
depth_values = np.fromstring(data_node.text.strip(), sep=' ')

# 데이터 크기가 width * height와 일치하는지 확인
if depth_values.size != width * height:
    raise ValueError(f"데이터 크기 불일치: 예상 {width * height}, 실제 {depth_values.size}")

# 데이터를 (height, width) 형태로 변환
depth_data = depth_values.reshape((height, width))

# 데이터 확인
print("Depth Map Shape:", depth_data.shape)
print("Depth Value Range:", depth_data.min(), "to", depth_data.max())

# Depth Map 시각화
depth_normalized = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX)
depth_normalized = np.uint8(depth_normalized)

plt.figure(figsize=(8, 6))
plt.imshow(depth_normalized, cmap='jet')
plt.colorbar(label="Depth Intensity")
plt.title("Depth Map Visualization")
plt.axis("off")
plt.savefig('png.png')
#plt.show()
