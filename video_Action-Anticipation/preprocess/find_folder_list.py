import os

def find_folders_with_keywords(base_folder, keywords):
    """
    주어진 상위 폴더(base_folder) 내의 폴더들 중에서,
    지정된 키워드 목록(keywords)이 포함된 폴더들을 찾고, 해당 폴더 경로 리스트를 반환.
    
    Args:
        base_folder (str): 탐색할 상위 폴더 경로
        keywords (list): 찾고자 하는 키워드 목록

    Returns:
        list: 키워드가 포함된 폴더 경로들의 리스트
    """
    matching_folders = []  # 조건을 만족하는 폴더들의 리스트

    # os.walk()을 사용하여 상위 폴더부터 모든 하위 폴더를 탐색
    for root, dirs, files in os.walk(base_folder):
        for folder_name in dirs:
            # 폴더 이름에 지정된 키워드 중 하나라도 포함되어 있는지 확인
            if any(keyword in folder_name for keyword in keywords):
                matching_folders.append(os.path.join(root, folder_name))  # 조건에 맞는 폴더 경로 추가

    return matching_folders


# 검색할 상위 폴더 경로와 키워드 목록 설정
base_folder = "/data/sophia/AVT/DATA/frames"  # 예시 경로를 설정
keywords = [
    'Making_a_cup_of_coffee_in_coffee_maker',
    'Making_a_cup_of_instant_coffee',
    'Stocking_up_pantry',
    'Using_handheld_smart_devices'
]

# 조건을 만족하는 폴더 경로 리스트 찾기
matching_folders = find_folders_with_keywords(base_folder, keywords)

# 결과 출력
print("Matching folders:")
for folder in matching_folders:
    print(folder)
