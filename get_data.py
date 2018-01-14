import os
import xml.etree.ElementTree as ET

def read_data(data_path):
    #general: lots of repeating code here with train, val, test but not sure how to easily make it better
    #we need the file with names of all training and validation images
    imgnames_path_train=os.path.join(data_path,'ImageSets','Main','train.txt')
    imgnames_path_val=os.path.join(data_path,'ImageSets','Main','val.txt')
    imgnames_path_test=os.path.join(data_path,'ImageSets','Main','test.txt')

    #then we need the path to the xmls with the annotations
    annot_path=os.path.join(data_path,'Annotations')
    #and finally the path to the jpeg ImageSets
    imgs_path=os.path.join(data_path,'JPEGImages')

    #we create a list with all filenames of the jpeg ImageSets
    train_files=[]
    val_files=[]
    test_files=[]

    with open(imgnames_path_train) as f:
        for line in f:
            train_files.append(line.strip()+'.jpg')
    with open(imgnames_path_val) as f:
        for line in f:
            val_files.append(line.strip()+'.jpg')
    with open(imgnames_path_test) as f:
        for line in f:
            test_files.append(line.strip()+'.jpg')

    #now we go trhough all annotation files and get the needed info from there
    annot_files=[os.path.join(annot_path,n) for n in os.listdir(annot_path)]

    #make list to save all annotation data
    anot_all=[]
    #make dict that maps class names to numbers (not sure if needed)
    map_classes=dict()
    #make dict that counts images per class
    count_classes=dict()


    #parse the xmls
    for annot in annot_files:
        tree=ET.parse(annot)
        root=tree.getroot()

        #get all objects
        objs=root.findall('object')
        #if there are objects in the picture (just doublecheck)
        if len(objs)>0:
            #get filename, width and height
            filename=root.find('filename').text
            width=int(root.find('size').find('width').text)
            height=int(root.find('size').find('height').text)
            data_cur={'filepath':os.path.join(annot_path,filename),'width':width,'height':height,'bboxes':[]}

            #iterate over the objects (can be several objects in 1 picture)
            for obj in objs:
                #get class, bounding box and difficulty
                class_name=obj.find('name').text
                bbox=obj.find('bndbox')
                x1=bbox.find('xmin').text
                y1=bbox.find('ymin').text
                x2=bbox.find('xmax').text
                y2=bbox.find('ymax').text
                #some objects are extra difficult and those are not counted in the official competition
                #if they are extra difficult, 'difficult'==1 else 'difficult'==0
                if obj.find('difficult') is not None:
                    diff=int(obj.find('difficult').text)==1
                data_cur['bboxes'].append({'x1':x1,'x2':x2,'y1':y1,'y2':y2,'difficult':diff,'class_name':class_name})

                #update the count of classes
                if class_name in count_classes:
                    count_classes[class_name]+=1
                else:
                    count_classes[class_name]=1
                #update the mapping from class names to numbers
                if class_name not in map_classes:
                    map_classes[class_name]=len(map_classes)

            #check to which set the image belongs
            #there are more annotations in the train-val set than there are names in the ImageSets. Hence, I put all files that are not found in any of the ImageSets in the training set.
            if filename in train_files:
                data_cur['Imageset']='train'
            elif filename in val_files:
                data_cur['Imageset']='val'
            elif filename in test_files:
                data_cur['Imageset']='test'
            else:
                data_cur['Imageset']='train'

            #append the information of the current picture to the total list
            anot_all.append(data_cur)



    return anot_all, count_classes,map_classes

data_path='Data/VOC2012'
print(read_data(data_path))
