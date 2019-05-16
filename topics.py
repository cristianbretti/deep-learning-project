from shutil import copyfile

if __name__ == "__main__":
    f = open('datasets/tiny-imagenet-200/wnids.txt', 'r')
    g = open('datasets/tiny-imagenet-200/words.txt', 'r')
    h = open('datasets/tiny-imagenet-200/val/val_annotations.txt', 'r')

    folder_names = f.readlines()
    words = g.readlines()
    val_annotations = h.readlines()

    # for i in range(20, len(folder_names)):
    for i in range(1):
        # folder = folder_names[i].strip()
        folder = 'n02129165'
        name = ''
        for w in words:
            w_folder_list = w.split()
            if w_folder_list[0] == folder:
                name = '-'.join(w_folder_list[1:])
                break
        count = 1
        for annotation in val_annotations:
            annotation_list = annotation.split()
            if annotation_list[1] == folder:
                copyfile('datasets/tiny-imagenet-200/val/images/' +
                         annotation_list[0], 'datasets/cat_val/' + name + '-' + str(count) + '.JPEG')
                count += 1
