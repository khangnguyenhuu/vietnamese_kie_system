def convert_xyminmax(list_box):
    new_list = []
    for box in list_box[0]["boundary_result"]:
        box = box[:-1]
        xmin = int(min(box[0::2])-5)
        xmax = int(max(box[0::2])+5)
        ymin = int(min(box[1::2])-5)
        ymax = int(max(box[1::2])+5)
        new_list.append([xmin, ymin, xmax, ymax])
    
    return new_list