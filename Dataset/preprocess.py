f = open("./all_bangla_chakma_v-2.txt")
bangla = []
chakma = []
i = 1
flip = 0
prev = ""
for lines in f:
    if flip == 0:
        line = lines.strip().split('bangla:')
        if(len(line) != 2):
            print(line,i,"bn")
        bangla.append(line[1])

    else :
        line = lines.strip().split('chakma:')
        if(len(line) != 2):
            print(line,i,"chk")
        chakma.append(line[1])

    if 'bangla:' in lines and 'bangla:' in prev:
        print(i,lines,prev)
    
    if 'chakma:' in lines and 'chakma:' in prev:
        print(i,"chk",lines,prev)
    
    flip ^= 1
    i += 1
    prev = lines

def create_csv(bangla,chakma):
    import csv

    # File path for the CSV file
    csv_file_path = 'bangla-chakma.csv'

    # Writing data to the CSV file
    with open(csv_file_path, mode='w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        
        # Write the headers
        headers = ['bangla', 'chakma']
        writer.writerow(headers)
        
        # Write the remaining rows
        for bangla_text, chakma_text in zip(bangla, chakma):
            writer.writerow([bangla_text, chakma_text])

    print(f"CSV file '{csv_file_path}' created successfully.")


create_csv(bangla,chakma)
