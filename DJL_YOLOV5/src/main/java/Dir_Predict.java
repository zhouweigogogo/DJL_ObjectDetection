import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.DetectedObjects.DetectedObject;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.modality.cv.translator.YoloV5Translator;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import org.opencv.core.*;
import org.opencv.core.Point;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.concurrent.ConcurrentHashMap;

import static utils.util.*;


public class Dir_Predict {

    private static float scale_w = 0f;
    private static float scale_h = 0f;
    private static final int[] input_shape = {640, 480};
    private static int errs = 0;

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void dir_predict(String dir_path, float nmsThres) {

        Map<String, Object> arguments = new ConcurrentHashMap<>();
        arguments.put("width", input_shape[0]);//图片以640宽度进行操作
        arguments.put("height", input_shape[1]);//图片以480高度进行操作
        arguments.put("resize", true);//调整图片大小
        arguments.put("rescale", true);//图片值编程0-1之间

        arguments.put("threshold", 0.2);//阈值小于0.2不显示
        arguments.put("nmsThreshold", nmsThres);

        /*
        0.3 : 1.3263157894736841
        0.25 : 1.0736842105263158
        0.2 : 1.0210526315789474
         */

        Translator<Image, DetectedObjects> translator = YoloV5Translator.builder(arguments).optSynsetArtifactName("class.names").build();
        Criteria<Image, DetectedObjects> criteria =
                Criteria.builder()
                        .setTypes(Image.class, DetectedObjects.class)
                        .optDevice(Device.cpu())
                        .optModelUrls("src/main/resources/yolov5")
                        .optModelName("best.torchscript")
                        .optTranslator(translator)
//                        .optProgress(new ProgressBar())
                        .optEngine("PyTorch")
                        .build();
        try (ZooModel<Image, DetectedObjects> model = ModelZoo.loadModel(criteria)) {

            // 多张图片
            File dir = new File(dir_path);
            // 指定文件夹的路径，自动扫描
//            List<String> image_names = new ArrayList<>();
//            getAllFile(dir, image_names);
            List<String> image_names = getImage_name();

            System.out.println("文件夹扫描完毕！共有：" + image_names.size() + " 张图片");

            for (String image_name : image_names) {

                Path path = Paths.get(dir_path + "/" + image_name);
                Image image = ImageFactory.getInstance().fromFile(path);

                scale_w = (float) (image.getWidth() / input_shape[0]);
                scale_h = (float) (image.getHeight() / input_shape[1]);

                DetectedObjects detection = detect(image, model);
                errs += Math.abs(detection.items().size() - getLabels(image_name));
                saveBoundingBoxImage(image, detection, image_name);
            }
            System.out.println("nmsThre: " + nmsThres + "\terrs: " + (float) errs / 190);
            System.out.println("预测完毕！");
            System.out.println("预测图片已保存在 bulid/output/ 路径下");

        } catch (RuntimeException | ModelException | TranslateException | IOException e) {
            e.printStackTrace();
        }

    }

    static DetectedObjects detect(Image img, ZooModel<Image, DetectedObjects> model) throws IOException, ModelNotFoundException, MalformedModelException, TranslateException {
        try (Predictor<Image, DetectedObjects> predictor = model.newPredictor()) {
            return predictor.predict(img);
        }
    }

    static Rect rect = new Rect();
    static Scalar color = new Scalar(0, 0, 255);

    private static void saveBoundingBoxImage(Image img, DetectedObjects detection, String image_name)
            throws IOException {
        Path outputDir = Paths.get("build/output");
        Files.createDirectories(outputDir);

        Path imagePath = outputDir.resolve(image_name);

        Mat mat = Image2Mat(img);
        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_BGR2RGB);

        String resultText = String.format("%s: %d", "Measurements", detection.items().size());
        System.out.println("image_name: " + image_name + "\tMeasurements: " + detection.items().size() + "\tLabel:" + getLabels(image_name));

        for (DetectedObject obj : detection.<DetectedObject>items()) {
            BoundingBox bbox = obj.getBoundingBox();
            Rectangle rectangle = bbox.getBounds();
            // 变换坐标
            double new_left_x = rectangle.getX() - rectangle.getWidth();
            double new_left_y = rectangle.getY() - rectangle.getHeight();
            double new_right_x = rectangle.getX() + rectangle.getWidth();
            double new_right_y = rectangle.getY() + rectangle.getHeight();

            rect.x = (int) (new_left_x * scale_w + new_right_x * scale_w) / 2;
            rect.y = (int) (new_left_y * scale_h + new_right_y * scale_h) / 2;
            rect.width = (int) (new_right_x * scale_w - new_left_x * scale_w) / 2;
            rect.height = (int) (new_right_y * scale_h - new_left_y * scale_h) / 2;

            // 画框
            Imgproc.rectangle(mat, rect, color, 2);
        }

        Imgproc.putText(mat, resultText,
                new Point(mat.width() - 425, mat.height() - 100),
                Imgproc.FONT_HERSHEY_COMPLEX,
                1.25,
                color);

        Imgcodecs.imwrite(imagePath.toString(), mat);
    }

}


