package org.manhdev;

import weka.core.Instances;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;

import java.awt.image.BufferedImage;
import java.io.File;
import javax.imageio.ImageIO;
import java.util.ArrayList;
import java.util.List;

public class DatasetCreator {
    // Tạo tập dữ liệu Weka từ danh sách ảnh và nhãn
    public Instances createDataset(List<String> imagePaths, List<String> labels) throws Exception {
        if (imagePaths.size() != labels.size()) {
            throw new IllegalArgumentException("The number of image paths and labels must be the same.");
        }

        // Khai báo các thuộc tính cho dataset
        ArrayList<Attribute> attributes = new ArrayList<>();

        // Thêm các thuộc tính vào dataset (ví dụ: 3 thuộc tính màu)
        attributes.add(new Attribute("Red"));
        attributes.add(new Attribute("Green"));
        attributes.add(new Attribute("Blue"));

        // Thêm thuộc tính lớp (các nhãn) vào dataset
        ArrayList<String> classValues = new ArrayList<>(labels);
        Attribute classAttribute = new Attribute("Class", classValues);
        attributes.add(classAttribute);

        // Tạo đối tượng Instances với tên dataset và thuộc tính
        Instances data = new Instances("MyDataset", attributes, 0);
        data.setClassIndex(data.numAttributes() - 1); // Thiết lập chỉ số lớp

        // Tạo các instance và thêm vào dataset
        for (int i = 0; i < imagePaths.size(); i++) {
            String imagePath = imagePaths.get(i);
            String label = labels.get(i);

            // Trích xuất đặc trưng từ ảnh
            double[] features = extractFeatures(imagePath);
            Instance instance = new DenseInstance(data.numAttributes());
            instance.setDataset(data);

            // Thiết lập giá trị cho các thuộc tính
            for (int j = 0; j < features.length; j++) {
                instance.setValue(attributes.get(j), features[j]);
            }

            // Đặt giá trị lớp cho instance
            instance.setClassValue(label);

            // Thêm instance vào dataset
            data.add(instance);
        }

        return data; // Trả về dataset đã tạo
    }

    private double[] extractFeatures(String imagePath) throws Exception {
        BufferedImage image = ImageIO.read(new File(imagePath));
        // Tăng cường dữ liệu: thêm các biến thể ảnh
        List<BufferedImage> augmentedImages = augmentImage(image);

        double[] totalFeatures = new double[3];
        for (BufferedImage augmentedImage : augmentedImages) {
            double[] features = calculateAverageColor(augmentedImage);
            totalFeatures[0] += features[0];
            totalFeatures[1] += features[1];
            totalFeatures[2] += features[2];
        }

        // Tính toán màu sắc trung bình từ tất cả các ảnh tăng cường
        int numImages = augmentedImages.size();
        totalFeatures[0] /= numImages;
        totalFeatures[1] /= numImages;
        totalFeatures[2] /= numImages;

        return totalFeatures; // Trả về mảng chứa các đặc trưng
    }

    private List<BufferedImage> augmentImage(BufferedImage image) {
        List<BufferedImage> augmentedImages = new ArrayList<>();
        augmentedImages.add(image); // Thêm ảnh gốc

        // Thêm một số biến thể ảnh
        augmentedImages.add(flipImage(image)); // Lật ảnh
        augmentedImages.add(rotateImage(image, 90)); // Xoay 90 độ
        augmentedImages.add(changeBrightness(image, 1.2f)); // Tăng độ sáng

        return augmentedImages;
    }

    private BufferedImage flipImage(BufferedImage image) {
        BufferedImage flipped = new BufferedImage(image.getWidth(), image.getHeight(), image.getType());
        for (int x = 0; x < image.getWidth(); x++) {
            for (int y = 0; y < image.getHeight(); y++) {
                flipped.setRGB(image.getWidth() - x - 1, y, image.getRGB(x, y));
            }
        }
        return flipped;
    }

    private BufferedImage rotateImage(BufferedImage image, double angle) {
        // Xoay ảnh
        int w = image.getWidth();
        int h = image.getHeight();
        BufferedImage rotated = new BufferedImage(w, h, image.getType());
        // Sử dụng Graphics2D để thực hiện xoay
        java.awt.Graphics2D g2d = rotated.createGraphics();
        g2d.rotate(Math.toRadians(angle), w / 2, h / 2);
        g2d.drawImage(image, 0, 0, null);
        g2d.dispose();
        return rotated;
    }

    private BufferedImage changeBrightness(BufferedImage image, float factor) {
        BufferedImage brightened = new BufferedImage(image.getWidth(), image.getHeight(), image.getType());
        for (int x = 0; x < image.getWidth(); x++) {
            for (int y = 0; y < image.getHeight(); y++) {
                int rgb = image.getRGB(x, y);
                int r = (int) Math.min((int)((rgb >> 16) & 0xFF) * factor, 255);
                int g = (int) Math.min((int)((rgb >> 8) & 0xFF) * factor, 255);
                int b = (int) Math.min((int)(rgb & 0xFF) * factor, 255);
                brightened.setRGB(x, y, (r << 16) | (g << 8) | b);
            }
        }
        return brightened;
    }

    private double[] calculateAverageColor(BufferedImage image) {
        double[] features = new double[3];
        int width = image.getWidth();
        int height = image.getHeight();
        long sumRed = 0, sumGreen = 0, sumBlue = 0;

        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                int rgb = image.getRGB(x, y);
                sumRed += (rgb >> 16) & 0xFF;
                sumGreen += (rgb >> 8) & 0xFF;
                sumBlue += rgb & 0xFF;
            }
        }

        double numPixels = width * height;
        features[0] = sumRed / numPixels;   // Màu đỏ trung bình
        features[1] = sumGreen / numPixels; // Màu xanh lá trung bình
        features[2] = sumBlue / numPixels;  // Màu xanh dương trung bình

        return features; // Trả về mảng chứa các đặc trưng
    }
}
