# For Byte files
def ByteResults():
    from byteModel.featureExtraction_Byte import featureExtration

    from byteModel.test_train_SplitByte import test_train_Split

    from byteModel.byteML import RandomModel, kNN_Model, LR_Model, RF_Model
    from byteModel.byteML import XgBoostModel, XgBoostBest_Model


    featureExtration()

    test_train_Split()

    RandomModel()
    kNN_Model()
    LR_Model()
    RF_Model()
    XgBoostModel()
    XgBoostBest_Model()


# For ASM files
def AsmResults():
    from asmModel.featureExtraction_ASM import featureExtraction

    from asmModel.test_train_SplitASM import test_train_Split

    from asmModel.asmML import RandomModel, kNN_Model, LR_Model, RF_Model
    from asmModel.asmML import XgBoostModel, XgBoostBest_Model

    featureExtration()

    test_train_Split()

    RandomModel()
    kNN_Model()
    LR_Model()
    RF_Model()
    XgBoostModel()
    XgBoostBest_Model()



# For Combined results
def MergedResults():
    from byte_ASM_Merge.merge import byte_asmMerge

    from byte_ASM_Merge.test_train_Split import split

    from byte_ASM_Merge.ML_Model import RF_Model, XgBoostModel, XgBoostBestModel

    byte_asmMerge()
    split()

    RF_Model()
    XgBoostModel()
    XgBoostBestModel()



# main funtion
if __name__ == '__main__':
    ByteResults()
    AsmResults()
    MergedResults()




