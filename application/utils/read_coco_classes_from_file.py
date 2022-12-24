def read_coco_classes_from_file(path='application/static/coco_classes.txt'):
    coco_classes_file = open(path, "r")

    coco_classes = coco_classes_file.read()
    coco_classes_file.close()
    return coco_classes.split(",")


