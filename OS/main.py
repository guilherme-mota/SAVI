#!/usr/bin/env python3


import os


def main():

    # Get the name of folders in a directory
    folders = os.listdir('/home/guilherme/SAVI_Datasets/rgbd-dataset')
    
    # Reorder folders in Alphabetical Order
    folders.sort()

    # Print results
    print(str(type(folders)) + '\n')
    print(str(type(folders[0])) + '\n')
    print('Folders: ' + str(folders))

    # Create txt file with code
    file = open("python_code.txt", "w")

    # Insert text in file
    for idx,folder in enumerate(folders):

        if idx == 0:
            file.write("if class_name == '" + folder + "':\r\n\tlabel = " + str(idx))
        else:
            file.write("\r\nelif class_name == '" + folder + "':\r\n\tlabel = " + str(idx))

    # Close file
    file.close()


if __name__ == "__main__":
    main()