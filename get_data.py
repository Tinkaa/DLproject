import os
import xml.etree.ElementTree as ET
import cv2

def read_data2(data_path):
    # general: lots of repeating code here with train, val, test but not sure how to easily make it better
    # we need the file with names of all training and validation images
    imgnames_path_train = os.path.join(data_path, 'ImageSets', 'Main', 'train.txt')
    imgnames_path_val = os.path.join(data_path, 'ImageSets', 'Main', 'val.txt')
    imgnames_path_test = os.path.join(data_path, 'ImageSets', 'Main', 'test.txt')

    # then we need the path to the xmls with the annotations
    annot_path = os.path.join(data_path, 'Annotations')
    # and finally the path to the jpeg ImageSets
    imgs_path = os.path.join(data_path, 'JPEGImages')

    # we create a list with all filenames of the jpeg ImageSets
    train_files = []
    val_files = []
    test_files = []

    with open(imgnames_path_train) as f:
        for line in f:
            train_files.append(line.strip() + '.jpg')
    with open(imgnames_path_val) as f:
        for line in f:
            val_files.append(line.strip() + '.jpg')
    try:
        with open(imgnames_path_test) as f:
            for line in f:
                test_files.append(line.strip() + '.jpg')
    except:
        pass

    # now we go trhough all annotation files and get the needed info from there
    annot_files = [os.path.join(annot_path, n) for n in os.listdir(annot_path)]

    # make list to save all annotation data
    anot_all = []
    # make dict that maps class names to numbers (not sure if needed)
    map_classes = dict()
    # make dict that counts images per class
    count_classes = dict()

    # parse the xmls
    for annot in annot_files:
        tree = ET.parse(annot)
        root = tree.getroot()

        # get all objects
        objs = root.findall('object')
        # if there are objects in the picture (just doublecheck)
        if len(objs) > 0:
            # get filename, width and height
            filename = root.find('filename').text
            width = int(root.find('size').find('width').text)
            height = int(root.find('size').find('height').text)
            data_cur = {'filepath': os.path.join(annot_path, filename), 'width': width, 'height': height, 'bboxes': []}

            # iterate over the objects (can be several objects in 1 picture)
            for obj in objs:
                # get class, bounding box and difficulty
                class_name = obj.find('name').text
                bbox = obj.find('bndbox')
                x1 = bbox.find('xmin').text
                y1 = bbox.find('ymin').text
                x2 = bbox.find('xmax').text
                y2 = bbox.find('ymax').text
                # some objects are extra difficult and those are not counted in the official competition
                # if they are extra difficult, 'difficult'==1 else 'difficult'==0
                if obj.find('difficult') is not None:
                    diff = int(obj.find('difficult').text) == 1
                data_cur['bboxes'].append(
                    {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'difficult': diff, 'class_name': class_name})

                # update the count of classes
                if class_name in count_classes:
                    count_classes[class_name] += 1
                else:
                    count_classes[class_name] = 1
                # update the mapping from class names to numbers
                if class_name not in map_classes:
                    map_classes[class_name] = len(map_classes)

            # check to which set the image belongs
            # there are more annotations in the train-val set than there are names in the ImageSets. Hence, I put all files that are not found in any of the ImageSets in the training set.
            if filename in train_files:
                data_cur['Imageset'] = 'train'
            elif filename in val_files:
                data_cur['Imageset'] = 'val'
            elif filename in test_files:
                data_cur['Imageset'] = 'test'
            else:
                data_cur['Imageset'] = 'train'

            # append the information of the current picture to the total list
            anot_all.append(data_cur)

    return anot_all, count_classes, map_classes


def read_data(input_path, train_classes=None):
    all_imgs = []

    classes_count = {}

    class_mapping = {}

    visualise = False

    # data_paths = [os.path.join(input_path, s) for s in ['VOC2007', 'VOC2012']]
    data_paths = [os.path.join(input_path, s) for s in ['VOC2012']]

    print('Parsing annotation files')

    for data_path in data_paths:

        annot_path = os.path.join(data_path, 'Annotations')
        imgs_path = os.path.join(data_path, 'JPEGImages')
        imgsets_path_trainval = os.path.join(data_path, 'ImageSets', 'Main', 'trainval.txt')
        imgsets_path_test = os.path.join(data_path, 'ImageSets', 'Main', 'test.txt')

        trainval_files = []
        test_files = []
        try:
            with open(imgsets_path_trainval) as f:
                for line in f:
                    trainval_files.append(line.strip() + '.jpg')
        except Exception as e:
            print(e)

        try:
            with open(imgsets_path_test) as f:
                for line in f:
                    test_files.append(line.strip() + '.jpg')
        except Exception as e:
            if data_path[-7:] == 'VOC2012':
                # this is expected, most pascal voc distibutions dont have the test.txt file
                pass
            else:
                print(e)

        annots = [os.path.join(annot_path, s) for s in os.listdir(annot_path)]
        idx = 0
        for annot in annots:
            try:
                idx += 1

                et = ET.parse(annot)
                element = et.getroot()

                element_objs = element.findall('object')
                element_filename = element.find('filename').text
                element_width = int(element.find('size').find('width').text)
                element_height = int(element.find('size').find('height').text)

                if len(element_objs) > 0:
                    annotation_data = {'filepath': os.path.join(imgs_path, element_filename), 'width': element_width,
                                       'height': element_height, 'bboxes': []}

                    if element_filename in trainval_files:
                        annotation_data['Imageset'] = 'train'
                    elif element_filename in test_files:
                        annotation_data['Imageset'] = 'test'
                    else:
                        annotation_data['Imageset'] = 'train'

                add_to_set = False
                for element_obj in element_objs:
                    class_name = element_obj.find('name').text
                    if train_classes is not None and class_name not in train_classes:
                        continue
                    add_to_set = True
                    if class_name not in classes_count:
                        classes_count[class_name] = 1
                    else:
                        classes_count[class_name] += 1

                    if class_name not in class_mapping:
                        class_mapping[class_name] = len(class_mapping)

                    obj_bbox = element_obj.find('bndbox')
                    x1 = int(round(float(obj_bbox.find('xmin').text)))
                    y1 = int(round(float(obj_bbox.find('ymin').text)))
                    x2 = int(round(float(obj_bbox.find('xmax').text)))
                    y2 = int(round(float(obj_bbox.find('ymax').text)))
                    difficulty = int(element_obj.find('difficult').text) == 1
                    annotation_data['bboxes'].append(
                        {'class': class_name, 'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'difficult': difficulty})
                if add_to_set:
                    all_imgs.append(annotation_data)

                if visualise:
                    img = cv2.imread(annotation_data['filepath'])
                    for bbox in annotation_data['bboxes']:
                        cv2.rectangle(img, (bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2']), (0, 0, 255))
                    cv2.imshow('img', img)
                    cv2.waitKey(0)

            except Exception as e:
                print(e)
                continue
    return all_imgs, classes_count, class_mapping

# data_path = 'Data/VOC2012'
# print(read_data(data_path))
