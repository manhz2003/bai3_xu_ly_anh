package org.manhdev;

import weka.core.Instances;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;

import java.util.Arrays;
import java.util.List;

public class Main {

    public static void main(String[] args) throws Exception {
        List<String> imagePaths = Arrays.asList(
                "/Users/nguyenthemanh/Downloads/hoa.jpeg",
                "/Users/nguyenthemanh/Downloads/dongvat1.jpg"
        );

        List<String> labels = Arrays.asList("flower", "animal");

        // Tạo tập dữ liệu từ ảnh
        DatasetCreator datasetCreator = new DatasetCreator();
        Instances dataset = datasetCreator.createDataset(imagePaths, labels);

        // Kiểm tra số lượng instance trong dataset
        System.out.println("Number of instances in dataset: " + dataset.numInstances());
        if (dataset.numInstances() == 0) {
            System.out.println("No instances in the dataset. Please check image paths or labels.");
            return;
        }

        // Đánh giá mô hình bằng cách sử dụng cross-validation
        ModelEvaluator evaluator = new ModelEvaluator();

        // 1. Phân loại và đánh giá bằng SVM
        SVMClassifier svmClassifier = new SVMClassifier();
        SMO svmModel = svmClassifier.trainSVM(dataset);
        System.out.println("Evaluating SVM using cross-validation...");
        evaluator.crossValidateModel(svmModel, dataset, 2);  // Giảm xuống 2-fold cross-validation

        // 2. Phân loại và đánh giá bằng KNN
        KNNClassifier knnClassifier = new KNNClassifier();
        IBk knnModel = knnClassifier.trainKNN(dataset, 3); // K=3
        System.out.println("Evaluating KNN using cross-validation...");
        evaluator.crossValidateModel(knnModel, dataset, 2);  // Giảm xuống 2-fold cross-validation

        // 3. Phân loại và đánh giá bằng Decision Tree
        DecisionTreeClassifier decisionTreeClassifier = new DecisionTreeClassifier();
        J48 treeModel = decisionTreeClassifier.trainDecisionTree(dataset);
        System.out.println("Evaluating Decision Tree using cross-validation...");
        evaluator.crossValidateModel(treeModel, dataset, 2);  // Giảm xuống 2-fold cross-validation
    }
}
