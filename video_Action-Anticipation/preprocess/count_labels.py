import os
from collections import defaultdict

def collect_label_info(src_folder_list, output_file):
    label_count = defaultdict(int)
    for src_folder in src_folder_list:
        # os.walk를 사용하여 폴더를 재귀적으로 탐색
        for root, dirs, files in os.walk(src_folder):
            print(root)
            for file in files:
                #print(file)
                # 파일 이름에서 레이블 이름 추출
                if file.endswith(".png"):
                    # 파일 이름이 [frame number]_[label name].png 형식이므로 "_"로 분할
                    label_name = "_".join(file.split("_")[1:]).replace(".png", "")
                    # 레이블 카운트를 1씩 증가
                    label_count[label_name] += 1

    # 결과를 txt 파일로 저장
    with open(output_file, 'w') as f:
        for label, count in label_count.items():
            f.write(f"{label}: {count} frames\n")
    
    print(f"Label information saved to {output_file}")

# 사용 예시
src_folder_list = [
    '/data/sophia/AVT/DATA/frames/14_3_cam_2_Stocking_up_pantry',
    '/data/sophia/AVT/DATA/frames/01_4_cam_2_Stocking_up_pantry',
    '/data/sophia/AVT/DATA/frames/13_2_cam_1_Using_handheld_smart_devices',
    '/data/sophia/AVT/DATA/frames/13_3_cam_1_Stocking_up_pantry',
    '/data/sophia/AVT/DATA/frames/13_3_cam_2_Stocking_up_pantry',
    '/data/sophia/AVT/DATA/frames/15_1_cam_1_Using_handheld_smart_devices',
    '/data/sophia/AVT/DATA/frames/18_1_cam_1_Using_handheld_smart_devices',
    '/data/sophia/AVT/DATA/frames/03_4_cam_1_Making_a_cup_of_coffee_in_coffee_maker',
    '/data/sophia/AVT/DATA/frames/15_2_cam_1_Using_handheld_smart_devices',
    '/data/sophia/AVT/DATA/frames/13_4_cam_1_Making_a_cup_of_coffee_in_coffee_maker',
    '/data/sophia/AVT/DATA/frames/08_3_cam_2_Making_a_cup_of_instant_coffee',
    '/data/sophia/AVT/DATA/frames/05_3_cam_2_Making_a_cup_of_instant_coffee',
    '/data/sophia/AVT/DATA/frames/01_4_cam_1_Making_a_cup_of_coffee_in_coffee_maker',
    '/data/sophia/AVT/DATA/frames/01_3_cam_2_Making_a_cup_of_instant_coffee',
    '/data/sophia/AVT/DATA/frames/03_3_cam_1_Making_a_cup_of_instant_coffee',
    '/data/sophia/AVT/DATA/frames/07_3_cam_2_Making_a_cup_of_instant_coffee',
    '/data/sophia/AVT/DATA/frames/08_3_cam_1_Making_a_cup_of_instant_coffee',
    '/data/sophia/AVT/DATA/frames/10_3_cam_1_Making_a_cup_of_instant_coffee',
    '/data/sophia/AVT/DATA/frames/10_3_cam_2_Making_a_cup_of_instant_coffee',
    '/data/sophia/AVT/DATA/frames/20_1_cam_2_Using_handheld_smart_devices',
    '/data/sophia/AVT/DATA/frames/01_4_cam_1_Stocking_up_pantry',
    '/data/sophia/AVT/DATA/frames/06_3_cam_1_Stocking_up_pantry',
    '/data/sophia/AVT/DATA/frames/06_3_cam_2_Stocking_up_pantry',
    '/data/sophia/AVT/DATA/frames/03_4_cam_2_Stocking_up_pantry',
    '/data/sophia/AVT/DATA/frames/18_1_cam_2_Using_handheld_smart_devices',
    '/data/sophia/AVT/DATA/frames/20_2_cam_1_Using_handheld_smart_devices',
    '/data/sophia/AVT/DATA/frames/14_3_cam_1_Stocking_up_pantry',
    '/data/sophia/AVT/DATA/frames/13_3_cam_1_Making_a_cup_of_instant_coffee',
    '/data/sophia/AVT/DATA/frames/05_3_cam_1_Making_a_cup_of_instant_coffee',
    '/data/sophia/AVT/DATA/frames/13_3_cam_2_Making_a_cup_of_instant_coffee',
    '/data/sophia/AVT/DATA/frames/13_4_cam_2_Making_a_cup_of_coffee_in_coffee_maker',
    '/data/sophia/AVT/DATA/frames/03_3_cam_2_Making_a_cup_of_instant_coffee',
    '/data/sophia/AVT/DATA/frames/03_4_cam_2_Making_a_cup_of_coffee_in_coffee_maker',
    '/data/sophia/AVT/DATA/frames/07_3_cam_1_Making_a_cup_of_instant_coffee',
    '/data/sophia/AVT/DATA/frames/15_2_cam_2_Using_handheld_smart_devices',
    '/data/sophia/AVT/DATA/frames/13_2_cam_2_Using_handheld_smart_devices',
    '/data/sophia/AVT/DATA/frames/15_1_cam_2_Using_handheld_smart_devices',
    '/data/sophia/AVT/DATA/frames/20_2_cam_2_Using_handheld_smart_devices',
    '/data/sophia/AVT/DATA/frames/20_1_cam_1_Using_handheld_smart_devices',
    '/data/sophia/AVT/DATA/frames/01_4_cam_2_Making_a_cup_of_coffee_in_coffee_maker',
]
output_file = '/data/sophia/AVT/DATA/output.txt'  # 결과를 저장할 txt 파일 경로

collect_label_info(src_folder_list, output_file)
