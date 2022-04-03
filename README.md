# vietnamese_kie_system
Key information extraction for Vietnamese documents

# Project struture:
/app:
    /libs: include all libs using in this project (MMOCR, vietocr)
    /routes: routing file, match url with function
    /src: predict source of 3 models
    /utils: tools
    class_list.txt
    download_weights.sh
    main.py
    
# Using

```bash
docker-compose build
docker-compose up
```

End-point: localhost:80/kie
Return: Json file with format
```json
{
    [
        "information field"
        [
            x1,
            y1,
            x2,
            y2,
            x3,
            y3
            x4,
            y4,
            "recognition string of this bounding box"
        ]
    ]
    .
    .
    .
}
```