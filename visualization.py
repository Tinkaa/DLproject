import re
import matplotlib.pyplot as plt


if __name__ == "__main__":
    file_list = ['train_3_classes', 'train_all_classes_1', 'train_3_classes_focal']
    name = ['3 classes', 'all classes', 'focal loss with 3 classes']
    colors = ['r', 'g', 'b']
    num_epochs = list(range(0, 120))
    fig_accuracy = plt.figure(4)
    for index, file in enumerate(file_list):
        mean_num_boxes = []
        classifier_accuracy = []
        loss_rpn_classifier = []
        loss_rpn_regression = []
        loss_detector_classifier = []
        loss_detector_regression = []
        with open(file + '.out') as fp:
            line = fp.readline()
            while line:
                if "Mean number of bounding boxes" in line:
                    num = re.findall("\d+\.\d+", line)
                    mean_num_boxes.append(float(num[0]))

                if "Classifier accuracy" in line:
                    num = re.findall("\d+\.\d+", line)
                    classifier_accuracy.append(float(num[0]))

                if "Loss RPN classifier" in line:
                    num = re.findall("\d+\.\d+", line)
                    loss_rpn_classifier.append(float(num[0]))

                if "Loss RPN regression" in line:
                    num = re.findall("\d+\.\d+", line)
                    loss_rpn_regression.append(float(num[0]))

                if "Loss Detector classifier" in line:
                    num = re.findall("\d+\.\d+", line)
                    loss_detector_classifier.append(float(num[0]))

                if "Loss Detector regression" in line:
                    num = re.findall("\d+\.\d+", line)
                    loss_detector_regression.append(float(num[0]))

                line = fp.readline()

        plt.figure(4)
        plt.plot(num_epochs, classifier_accuracy, colors[index], label=name[index])

        fig = plt.figure(index)
        plt.plot(num_epochs, loss_rpn_classifier, 'g', label="Loss RPN classifier")
        plt.plot(num_epochs, loss_rpn_regression, 'b', label="Loss RPN regression")
        plt.plot(num_epochs, loss_detector_classifier, 'c', label="Loss detector classifier")
        plt.plot(num_epochs, loss_detector_regression, 'm', label="Loss detector regression")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc=0)
        plt.title(file)
        fig.savefig('./figs/' + file + '.jpg')

    plt.figure(4)
    plt.legend(loc=0)
    plt.title('Classifier accuracy from RPN for 3 setups')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    fig_accuracy.savefig('./figs/accuracy.jpg')
