f = open("./chatgaiya-v4.txt")
f1 = open("./chittagonian.txt","w")

for line in f:
    lines = line.strip().split('\t')
    f1.writelines("bangla:"+lines[0]+"\n")
    f1.writelines("chatgaiya:"+lines[1]+"\n")