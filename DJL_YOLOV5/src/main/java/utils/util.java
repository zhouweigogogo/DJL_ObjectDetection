package utils;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import org.jetbrains.annotations.NotNull;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;


public class util {

    public static void main(String[] args) throws IOException {
//        for (String name : getImage_name()) {
//            System.out.println(name);
//        }
//        System.out.println(getLabels("ch01_20221025135529_timingCap.jpg"));
    }

    public static Image mat2Image(Mat mat) {
        return ImageFactory.getInstance().fromImage(HighGui.toBufferedImage(mat));
    }

    // 返回的是BGR格式图像
    public static Mat Image2Mat(Image image) throws IOException {
        // 将Image转化为NDArray
        NDManager manager = NDManager.newBaseManager();
        NDArray nd = image.toNDArray(manager);

        ByteBuffer bb = nd.toByteBuffer();

        Mat mat = new Mat((int) nd.getShape().get(0), (int) nd.getShape().get(1), CvType.CV_8UC3);
        mat.put(0, 0, bb.array());
        return mat;
    }


    public static NDArray getAnchors() throws IOException {
        Path path = Paths.get("src/main/resources/yolov5/yolo_anchors.txt");
        String data = Files.readString(path);
        String[] anchor = data.split(",");
        int[] temp = new int[anchor.length];
        NDManager manager = NDManager.newBaseManager();
        // 转成int
        for (int i = 0; i < anchor.length; i++) {
            temp[i] = Integer.parseInt(anchor[i].strip());
        }
        NDArray arr = manager.create(temp);
        arr = arr.reshape(new Shape(-1, 2));
        return arr;
    }

    public static int getLabels(String image_name) throws IOException {
        Path path = Paths.get("src/main/resources/yolov5/labels_count.txt");
        List<String> labels = Files.readAllLines(path);
        for (String label : labels) {
            String target = label.split(" ")[0];
            int count = Integer.parseInt(label.split(" ")[1]);
            if (target.equals(image_name))
                return count;
        }
        return 0;
    }

    public static List<String> getImage_name() throws IOException {
        Path path = Paths.get("src/main/resources/yolov5/labels_count.txt");
        List<String> labels = Files.readAllLines(path);
        List<String> image_names = new ArrayList<>();
        for (String label : labels) {
            String image_name = label.split(" ")[0];
            image_names.add(image_name);
        }
        return image_names;
    }

    public static NDArray sigmoid(@NotNull NDArray value) {
        return value.exp().div(value.exp().add(1));
    }

    public static void getAllFile(File fileInput, List<String> allFileList) {
        // 获取文件列表
        try {
            File[] fileList = fileInput.listFiles();
            assert fileList != null;
            for (File file : fileList) {
                if (file.isDirectory()) {
                    // 递归处理文件夹
                    // 如果不想统计子文件夹则可以将下一行注释掉
//                getAllFile(file, allFileList);
                } else {
                    // 如果是文件则将其加入到文件数组中
                    allFileList.add(file.getName());
                }
            }
        }catch (NullPointerException e){
            e.printStackTrace();
            System.out.println("请输入需要预测图片的文件夹的正确路径！");
        }
    }
}

