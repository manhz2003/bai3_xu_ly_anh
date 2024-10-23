package org.manhdev;

import weka.classifiers.lazy.IBk;
import weka.core.Instances;

public class KNNClassifier {

    public IBk trainKNN(Instances trainData, int k) throws Exception {
        IBk knn = new IBk(k);
        knn.buildClassifier(trainData);
        return knn;
    }
}
