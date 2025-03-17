import os

# 레이블 이름과 인덱스를 정의 (0~26까지 할당)
labels = {
    "Prepare_kitchen_appliance": 0,
    "Prep_ingredients": 1,
    "Mix_ingredients": 2,
    "Take_out_Kitchen_and_cooking_tools_": 3,
    "Using_Smartphone": 4,
    "Take_out_smartphone": 5,
    "Put_away_items__or_bags": 6,
    "Carrying_bags": 7,
    "Prepare_kitchen_appliances": 8,
    # "Playing_on_TV": 9,
    # "Playing_on_computer": 10,
    # "Exercise_with_instructions": 11,
    # "Watch_": 12,
    #"Reading_on_couch": 13,
    #"Reading_at_desk": 14,
    #"Reading_instruction": 15,
    #"Rest": 16,
    #"Prepare_for_activity": 17,
    #"Exercise_free_style": 18,
    #"Carrying_Big_object": 19,
    #"Carrying_Heavy_object": 20,
    #"Writing_on_paper_": 21,
    #"Typing_": 22,
    #"Taking_a_Quiz": 23,
    #"Virtual_meeting": 24,
    #"Carrying_Small_object": 25,
    #"Carrying_Light_object": 26,
}

def create_labels_txt(src_folder):
    # os.walk로 폴더 구조 탐색
    for root, dirs, files in os.walk(src_folder):
        label_file_path = os.path.join(root, "labels.txt")
        processed_numbers = set()  # 이미 처리된 번호를 저장하는 집합

        # PNG 파일을 찾기
        png_files = sorted([file for file in files if file.endswith(".png")])
        
        if png_files:
            # labels.txt 파일을 작성
            with open(label_file_path, 'w') as label_file:
                for png_file in png_files:
                    # 파일명에서 프레임 번호와 레이블 이름 추출
                    frame_number = png_file.split("_")[0]
                    label_name = "_".join(png_file.split("_")[1:]).replace(".png", "")

                    # # 이미 처리한 프레임 번호인지 확인
                    # if frame_number not in processed_numbers and label_name in labels:
                    #     # 해당 레이블에 맞는 번호를 파일에 작성
                    #     label_file.write(f"{labels[label_name]}\n")
                    #     processed_numbers.add(frame_number)  # 처리된 번호로 추가

                    # 이미 처리한 프레임 번호인지 확인
                    if label_name in labels:
                        # 해당 레이블에 맞는 번호를 파일에 작성
                        label_file.write(f"{labels[label_name]}\n")
                        processed_numbers.add(frame_number)  # 처리된 번호로 추가
                    else:
                        print(label_name)
                        label_file.write("9\n")
                        
            print(f"Labels written to {label_file_path}")

# 사용 예시
#src_folder = '/data/sophia/AVT/DATA/frames'  # frames 폴더 경로
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
    '/data/sophia/AVT/DATA/frames/01_3_cam_1_Making_a_cup_of_instant_coffee',
    '/data/sophia/AVT/DATA/frames/01_3_cam_2_Making_a_cup_of_instant_coffee',
]
for src_folder in src_folder_list:
    create_labels_txt(src_folder)
