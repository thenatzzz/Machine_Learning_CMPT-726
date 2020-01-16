from tabulate import tabulate
import os
import natsort
FILE_NAME = 'image_score_html.html'
IMAGE_FOLDER = 'sample_test_image'

def create_table_html(input_filename_list):
    list = []
    NUM_IMAGE = len(input_filename_list)
    pic_per_row = 13 #14
    num_row = 200 #715

    count =0
    for i in range(0,num_row):
        list_per_row = []
        for j in range(0,pic_per_row):
            if count == NUM_IMAGE:
                break
            pic = "<img src="+IMAGE_FOLDER+"/"+input_filename_list[count]+">"
            list_per_row.append(pic)
            count += 1
        list.append(list_per_row)
    return list

if __name__ == '__main__':
    input_file_name_list = os.listdir(IMAGE_FOLDER)
    input_file_name_list=natsort.natsorted(input_file_name_list,reverse=False)
    table = create_table_html(input_file_name_list)
    html_output = tabulate(table, tablefmt='html')

    with open(FILE_NAME,'w') as file:
        file.write(html_output)
