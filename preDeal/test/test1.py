with open('a.txt', 'r', encoding='utf-8') as f:
    print(f.readlines())
    f.seek(0)
    for i, x in enumerate(f.readlines()):
        print(i, ':', x, end='')
