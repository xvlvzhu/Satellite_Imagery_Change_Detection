with open("train.txt","w") as f:
    for i in range(0,20):
        for j in range(0,59):
            str0 = "originalData_256/" + "2015_" + str(i) +"_" + str(j) + "_256_.jpg"
            str1 = "originalLabel_256/" + "2015_" + str(i) +"_" + str(j) + "_256_.jpg"
            f.write(str0+" "+str1+'\n')
    for i in range(0,20):
        for j in range(0,59):
            str0 = "originalData_256/" + "2017_" + str(i) +"_" + str(j) + "_256_.jpg"
            str1 = "originalLabel_256/" + "2017_" + str(i) +"_" + str(j) + "_256_.jpg"
            f.write(str0+" "+str1+'\n')


