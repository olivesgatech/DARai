import os

# 레이블 목록을 딕셔너리로 변환
label_list = [
    "Add_coffee", "Add_water", "Boxing_workout", "Conversation_on_the_phone", #3
    "Fill_coffee_machine_with_water", "Fill_kettle_with_water", "Get_cup", #6
    "Get_filter", "Get_instant_coffee_", "Get_magazine/book", "Get_marker", #10
    "Get_paper", "Get_paper/book", "Get_pen", "Get_spoon", "Jumping_jacks", #15
    "Jumping_rope", "Lay_down", "Open_software/platform", "Pick_up_box", #19
    "Place_cup", "Place_filter", "Play_news", "Play_sport", "Playing_reaction_game", #24
    "Playing_speed_game", "Put_down_box", "Put_in_cabinets", "Read", "Read_magazine/book", #29 
    "Read_paper/book", "Read_poster_on_the_wall", "Scroll_on_the_phone", #32
    "Simple_Jumping", "Squat", "Stir_", "Take_out_items_from_bag", "Turn_on_kettle", #37
    "Type", "Type_on_external_keyboard", "Type_on_laptop_keyboard", "Use_blanket", #41
    "Wake_up", "Walk_with_box", "Yoga_stretch", "put_down_bags", "walk_with_bags", "None" #47
]

# 레이블과 인덱스의 딕셔너리 생성
label_dict = {label: idx for idx, label in enumerate(label_list)}

# l3.txt 파일 읽고 레이블 인덱스를 추가하는 코드
def append_labels_to_file(input_file, label_dict):
    output_file = input_file.replace("l3.txt", "l3_with_labels.txt")
    
    with open(input_file, 'r') as file:
        lines = file.readlines()
    
    # 각 줄의 레이블을 찾아 인덱스를 추가
    with open(output_file, 'w') as new_file:
        for line in lines:
            # 라인에서 현재 레이블을 추출 (예: Prepare_kitchen_appliances)
            parts = line.strip().split()
            if len(parts) >= 2:
                label_name = parts[1]  # 두 번째 요소가 레이블 이름
                # 해당 레이블의 인덱스를 찾기
                label_index = label_dict.get(label_name, "None")  # 레이블이 없으면 None 반환
                # 인덱스를 추가하여 파일에 작성
                new_line = f"{line.strip()} {label_index}\n"
                new_file.write(new_line)
            else:
                # 만약 올바르지 않은 라인 형식이라면 그대로 복사
                new_file.write(line)

root_directory = "/data/sophia/AVT/DATA/frames/"

# os.walk를 사용하여 하위 폴더를 모두 탐색
for root, dirs, files in os.walk(root_directory):
    for file_name in files:
        if file_name == "l3.txt":
            # l3.txt 파일을 찾으면 레이블을 추가하는 함수 호출
            full_path = os.path.join(root, file_name)
            print(f"Found l3.txt: {full_path}")
            append_labels_to_file(full_path, label_dict)