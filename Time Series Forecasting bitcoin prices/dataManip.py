import numpy as np
import csv

class Data(object):
    # NOTE: interperates the file values for the neural net's weights
    def unpackMemory(self, dir, memFile):
        file = open(dir+memFile)
        weights = csv.reader(file)
        list_of_weights = list(weights)
        data = []
        for index in range(int(len(list_of_weights)/2)):
            fullTempArr = []
            tempArr0 = []
            tempArr1 = []
            for num in list_of_weights[2*index]:
                try:
                    tempArr0.append(float(num))
                except:
                    tempArr0.append(str(num))
            for index in tempArr0:
                if index == 'zzz':
                    fullTempArr.append(tempArr1)
                    tempArr1 = []
                else:
                    tempArr1.append(index)
            data.append(fullTempArr)
        return np.asarray(data)

    # NOTE: saves weights in to the csv file
    def packMemory(self, weights):
        # FIXME: change the data type of the 'strength_of_connection' variable
        strength_of_connection = weights
        with open(self.filename, 'a') as data:
            writer = csv.writer(data)
            for item in strength_of_connection:
                tempArr = []
                _row, _col = item.shape
                for x in range(_row):
                    for y in range(_col):
                        tempArr.append(item[x, y])
                    tempArr.append('zzz')
                writer.writerow(tempArr)

    # NOTE: unpack data needed to begin trainning
    def unpackData(self, dir, dumpFile):
            data = []
            tempArr0 = []
            moldData = []
            file = open(dir+dumpFile)
            dataInFile = csv.reader(file)
            listData = list(dataInFile)
            for i in range(len(listData)):
                if i%2 == 0:
                    data.append(listData[i])
            for d in range(len(data)):
                for item in data[d]:
                    try:
                        if d == 0:
                            tempArr0.append(int(item))
                        else:
                            tempArr0.append(float(item))
                    except:
                        tempArr0.append(str(item))
                moldData.append(tempArr0);
                tempArr0 = []
            return moldData

    # NOTE: packs the data in the original file
    def packData(self, data, dir, dumpFile):
        with open(dir+dumpFile, 'w') as file:
            writer = csv.writer(file)
            for item in range(0, len(data)):
                print("Completed: "+str(item)+'/'+str(len(data)))
                writer.writerow(data[item])
                print("Completed: "+str(item+1)+'/'+str(len(data)))
        print("Done!!!")
