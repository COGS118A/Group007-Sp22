import os

def get_files(file_path):
    return os.listdir(file_path)

def remove_ext(file_list):
    return [file.split('.')[0] for file in file_list]

def delete_from_list(list1, list2):
    return [elem for elem in list1 if elem not in list2]

def main():
    label_file_path = "bdd100k/labels/box_track_20/train/"
    img_file_path = "data/bdd/images/track/train/"
    label_files_temp = get_files(label_file_path)
    label_files = remove_ext(label_files_temp)
    img_files = get_files(img_file_path)
    names_to_delete = delete_from_list(label_files, img_files)
    print(len(names_to_delete))  # 1200
    for name in names_to_delete:
        os.remove(label_file_path + name + ".json")

if __name__ == '__main__':
    main()