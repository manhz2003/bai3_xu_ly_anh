package org.manhdev;

import weka.classifiers.trees.J48;
import weka.core.Instances;

public class DecisionTreeClassifier {

    public J48 trainDecisionTree(Instances trainData) throws Exception {
        J48 tree = new J48();
        tree.buildClassifier(trainData);
        return tree;
    }
}
