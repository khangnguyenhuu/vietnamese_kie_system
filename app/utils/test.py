import json
with open("gt.json", encoding="utf-8") as fp1:
    data = json.load(fp1)
    for i in data:
        print(data[i]["result"])