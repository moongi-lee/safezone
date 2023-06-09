# tracking

import requests

detect_txt_file = 'C:/Users/leeyo/Project/safezone/safezone/media/yolov5/runs/detect/exp/video_test.txt'
f = open(detect_txt_file,'r')
text_data = f.read()
f.close()

print(text_data)


# discord 연동
# discord = "https://discord.com/api/webhooks/1112547580702888066/vE27sKAJFlsHgR4v-shbzPKRK7CYJBqp3e3sk6f1OUJx0bdtTvcgjnaQVAq5yIMe0HU3"

# discord_headers={
#     'Content-Type':'application/json'
# }

# discord_data = '{"content":"discord test"}'
# response = requests.post(discord, headers=discord_headers, data=discord_data)



# count = {'0' : 1, '1' : 2, '2' : 0, '3' : 0, '4' : 5, '5' : 0}
# string_test = '17:1 2 4 '
# string_test_split = string_test.split(':')[1].split(' ')

# print(string_test_split)
# for count_index in count:                
#     if count_index in string_test_split:
#         count[count_index] += 1
#     else:
#         count[count_index] = 0
                
# print(count)

# test = {'0' : 1, '1' : 2, '2' : 3, '3' : 4, '4' : 5, '5' : 6}
# for test_check in test:
#     print(test_check)
# tracking = subprocess.check_output('python C:/Users/leeyo/Project/yolov5/detect.py --weights C:/Users/leeyo/Project/yolov5/runs/train/yolov5s_third/weights/best.pt --source C:/Users/leeyo/Project/yolov5/runs/detect/exp/video_test.mp4 --save-txt ', shell=True, universal_newlines=True)

# print('tracking = ', tracking)