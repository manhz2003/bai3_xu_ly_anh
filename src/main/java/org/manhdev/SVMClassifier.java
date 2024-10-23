package org.manhdev;

import weka.classifiers.functions.SMO;
import weka.core.Instances;

public class SVMClassifier {

    public SMO trainSVM(Instances trainData) throws Exception {
        SMO svm = new SMO();
        svm.buildClassifier(trainData);
        return svm;
    }
}
