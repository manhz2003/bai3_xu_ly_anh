package org.manhdev;

import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.classifiers.Evaluation;

import java.util.Random;

public class ModelEvaluator {

    // Phương thức để đánh giá mô hình bằng cross-validation
    public void crossValidateModel(Classifier model, Instances dataset, int folds) throws Exception {
        Evaluation evaluation = new Evaluation(dataset);
        evaluation.crossValidateModel(model, dataset, folds, new Random(1));

        // In kết quả cho từng lớp
        int numClasses = dataset.numClasses();
        System.out.println("Accuracy: " + evaluation.pctCorrect() + "%");
        for (int i = 0; i < numClasses; i++) {
            System.out.println("Class " + dataset.classAttribute().value(i));
            System.out.println("Precision: " + evaluation.precision(i));
            System.out.println("Recall: " + evaluation.recall(i));
            System.out.println("F1 Score: " + evaluation.fMeasure(i));
        }
        System.out.println("Time taken: " + evaluation.totalCost());
    }

}
