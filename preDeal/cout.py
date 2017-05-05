# 拷贝文件
with open("D://data/train.csv", 'r') as file,  open("some_train.csv", "w") as write_file:
    all_lines = file.readlines()
    for i in range(500):
        write_file.write(all_lines[i])

print("done!")
