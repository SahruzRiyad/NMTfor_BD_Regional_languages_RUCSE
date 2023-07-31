f = open("./all_bangla_chakma_v-2.txt","r")
num_bn = 0
num_ct = 0

mx_bn = 0
mn_bn = 1000

mx_ct = 0
mn_ct = 1000

total_words_bn = 0
total_words_ct = 0

unique_words_bn = 0
unique_words_ct = 0

uni_list_bn = []
uni_list_ct = []

total_char_len_bn = 0
total_char_len_ct = 0

i = 0
flag = 0
prev = ""

for line in f:
    if "bangla" in line:
        num_bn = num_bn + 1

        mx_bn = max(mx_bn,len(line))
        mn_bn = min(mn_bn,len(line))

        total_char_len_bn = total_char_len_bn + len(line) - 7

        sep = line.strip().split(' ')

        for x in sep:
            if(len(x) >= 2):
                total_words_bn = total_words_bn + 1

                if x not in uni_list_bn:
                    unique_words_bn = unique_words_bn + 1

                uni_list_bn.append(x)

    elif "chakma" in line:
        num_ct = num_ct + 1

        mx_ct = max(mx_ct,len(line))
        mn_ct = min(mn_ct,len(line))

        sep = line.strip().split(' ')

        total_char_len_ct = total_char_len_ct + len(line) - 7


        for x in sep:
            if(len(x) >= 2):
                total_words_ct = total_words_ct + 1

                if x not in uni_list_ct:
                    unique_words_ct = unique_words_ct + 1
                    
                uni_list_ct.append(x)

        flag = 0
    
    i += 1
    prev = line

print(num_bn,mx_bn,mn_bn,total_words_bn,unique_words_bn,total_char_len_bn/1024)
print(num_ct,mx_ct,mn_ct,total_words_ct,unique_words_ct,total_char_len_ct/1024)
