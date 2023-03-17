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

import javax.imageio.IIOException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;
import java.util.Scanner;
import java.util.concurrent.ConcurrentHashMap;

public class Sig_Predict {

    private static float scale_w = 0f;
    private static float scale_h = 0f;
    private static final int[] input_shape = {640, 480};

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) {
        System.out.println("请输入需要预测图片的路径：例如，img/123.jpg");
        Scanner scan = new Scanner(System.in);
        String image_path = scan.nextLine();
        scan.close();

        sig_redict(image_path);
        System.out.println("预测完毕！");
        System.out.println("预测图片已保存为 bulid/output/detected.png");
    }

    public static void sig_redict(String image_path) {
        Map<String, Object> arguments = new ConcurrentHashMap<>();
        arguments.put("width", input_shape[0]);//图片以640宽度进行操作
        arguments.put("height", input_shape[1]);//图片以480高度进行操作
        arguments.put("resize", true);//调整图片大小
        arguments.put("rescale", true);//图片值编程0-1之间

        arguments.put("threshold", 0.2);//阈值小于0.2不显示
        arguments.put("nmsThreshold", 0.2);
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
                        .optModelUrls("/yolov5")
                        .optModelName("best.torchscript")
                        .optTranslator(translator)
//                        .optProgress(new ProgressBar())
                        .optEngine("PyTorch")
                        .build();
        try (ZooModel<Image, DetectedObjects> model = ModelZoo.loadModel(criteria)) {
            Path path = Paths.get(image_path);
            try {
//                System.out.println(path);
                Image image = ImageFactory.getInstance().fromFile(path);
                scale_w = (float) (image.getWidth() / input_shape[0]);
                scale_h = (float) (image.getHeight() / input_shape[1]);
                DetectedObjects detection = detect(image, model);
                saveBoundingBoxImage(image, detection);
                System.out.println("预测完毕！");
                System.out.println("预测图片已保存为 bulid/output/detected.png");

            } catch (IIOException e) {
                e.printStackTrace();
                System.out.println("请输入需要预测图片的正确路径！");
            }


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

    private static void saveBoundingBoxImage(Image img, DetectedObjects detection)
            throws IOException {
        Path outputDir = Paths.get("build/output");
        Files.createDirectories(outputDir);

        Path imagePath = outputDir.resolve("detected.png");

        Mat mat = utils.util.Image2Mat(img);
        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_BGR2RGB);

        String resultText = String.format("%s: %d", "Measurements", detection.items().size());
        System.out.println("Measurements: " + detection.items().size());


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


