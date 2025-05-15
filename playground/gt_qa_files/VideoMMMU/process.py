import json

choices=["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

with open('adaption.json','r') as file:
    data=json.load(file)
    
with open('comprehension.json','r') as file:
    data2=json.load(file)
    
with open('perception.json','r') as file:
    data3=json.load(file)
    
data=data+data2+data3
    
for idx in range(len(data)):
    options=data[idx]['options'].replace("\n","").split("' '")
    options[0]=options[0][2:]
    options[-1]=options[-1][:-2]
    for i in range(len(options)):
        options[i]=choices[i]+". "+options[i]
    data[idx]['choices']=options
    data[idx]['answer_number']=choices.index(data[idx]['answer'])
print(len(data))
print(data[50])

with open ('total.json','w',encoding='utf-8') as file:
    json.dump(data,file,indent=4)