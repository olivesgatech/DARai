import os

# 폴더 리스트
folder_list = [
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
    '/data/sophia/AVT/DATA/frames/01_3_cam_1_Making_a_cup_of_instant_coffee',
    '/data/sophia/AVT/DATA/frames/01_3_cam_2_Making_a_cup_of_instant_coffee',
]

# 지정된 폴더들에서 labels.txt 파일을 찾고, 8을 0으로 변경
for folder in folder_list:
    labels_path = os.path.join(folder, 'labels.txt')
    
    if os.path.exists(labels_path):
        # labels.txt 파일 읽기
        with open(labels_path, 'r') as file:
            lines = file.readlines()

        # 모든 8을 0으로 바꾸기
        modified_lines = [line.replace('8', '0') for line in lines]

        # 수정된 내용을 labels.txt에 다시 쓰기
        with open(labels_path, 'w') as file:
            file.writelines(modified_lines)

        print(f"Modified {labels_path}")
    else:
        print(f"labels.txt not found in {folder}")
