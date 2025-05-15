import json

with open('/mnt/cloud_disk/jinhongbo/LLaVA-NeXT/playground/gt_qa_files/VideoMME/short_val_qa.json','r',encoding='utf-8') as file:
    data1=json.load(file)
    
print(len(data1))

with open('/mnt/cloud_disk/jinhongbo/LLaVA-NeXT/playground/gt_qa_files/VideoMME/medium_val_qa.json','r',encoding='utf-8') as file:
    data2=json.load(file)
    
print(len(data2))

with open('/mnt/cloud_disk/jinhongbo/LLaVA-NeXT/playground/gt_qa_files/VideoMME/long_val_qa.json','r',encoding='utf-8') as file:
    data3=json.load(file)
    
print(len(data3))

all_data=data1+data2+data3

print(len(all_data))

with open('/mnt/cloud_disk/jinhongbo/LLaVA-NeXT/playground/gt_qa_files/VideoMME/val_qa.json','w') as file:
    json.dump(all_data,file,indent=4)
    