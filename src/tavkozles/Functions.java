package tavkozles;
import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import java.lang.Math;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.opencv.core.Core.addWeighted;
import static org.opencv.core.CvType.*;
import static org.opencv.highgui.HighGui.imshow;
import static org.opencv.highgui.HighGui.waitKey;
import static org.opencv.imgcodecs.Imgcodecs.imread;
import static org.opencv.imgcodecs.Imgcodecs.imwrite;
import static org.opencv.imgproc.Imgproc.*;
import org.opencv.imgcodecs.Imgcodecs;

public  class Functions {
    public static final int row = 0, col = 0;

    public static void intro(){
        Mat im = imread("mayans.jpg",1);
        imshow("kep", im);
        waitKey(0);
    }

    public static void lab01(){
        Mat im1 = imread("3.jpg", 1);
        Mat im2 = imread("5.jpg", 1);
        imshow("Film", im1);
        waitKey(0);
        Mat im3 = im2.clone();
        for (float q = 0; q < 1.01; q += 0.02f)
        {
            addWeighted(im1, 1.0f - q, im2, q, 0, im3);
            imshow("Film", im3);
            waitKey(100);
            if (q < 0.67 && q>0.65) imwrite("keverek.bmp", im3);
        }
        waitKey(0);
    }

    public static Mat GuidedImageFilter(Mat I, Mat p, int r, double eps) {
        I.convertTo(I, CvType.CV_64FC1);
        p.convertTo(p, CvType.CV_64FC1);
        //[hei, wid] = size(I);
        int rows = I.rows();
        int cols = I.cols();
        // N = boxfilter(ones(hei, wid), r); % the size of each local patch; N=(2r+1)^2 except for boundary pixels.
        Mat N = new Mat();
        Imgproc.boxFilter(Mat.ones(rows, cols, I.type()), N, -1, new Size(r, r));
        // mean_I = boxfilter(I, r) ./ N;
        Mat mean_I = new Mat();
        Imgproc.boxFilter(I, mean_I, -1, new Size(r, r));
        // mean_p = boxfilter(p, r) ./ N
        Mat mean_p = new Mat();
        Imgproc.boxFilter(p, mean_p, -1, new Size(r, r));
        // mean_Ip = boxfilter(I.*p, r) ./ N;
        Mat mean_Ip = new Mat();
        Imgproc.boxFilter(I.mul(p), mean_Ip, -1, new Size(r, r));
        // cov_Ip = mean_Ip - mean_I .* mean_p; % this is the covariance of (I, p) in each local patch.
        Mat cov_Ip = new Mat();
        Core.subtract(mean_Ip, mean_I.mul(mean_p), cov_Ip);
        // mean_II = boxfilter(I.*I, r) ./ N;
        Mat mean_II = new Mat();
        Imgproc.boxFilter(I.mul(I), mean_II, -1, new Size(r, r));
        // var_I = mean_II - mean_I .* mean_I;
        Mat var_I = new Mat();
        Core.subtract(mean_II, mean_I.mul(mean_I), var_I);
        // a = cov_Ip ./ (var_I + eps); % Eqn. (5) in the paper;
        Mat a = new Mat();
        Core.add(var_I, new Scalar(eps), a);
        Core.divide(cov_Ip, a, a);
        //b = mean_p - a .* mean_I; % Eqn. (6) in the paper;
        Mat b = new Mat();
        Core.subtract(mean_p, a.mul(mean_I), b);
        // mean_a = boxfilter(a, r) ./ N;
        Mat mean_a = new Mat();
        Imgproc.boxFilter(a, mean_a, -1, new Size(r, r));
        Core.divide(mean_a, N, mean_a);
        // mean_b = boxfilter(b, r) ./ N;
        Mat mean_b = new Mat();
        Imgproc.boxFilter(b, mean_b, -1, new Size(r, r));
        Core.divide(mean_b, N, mean_b);
        // q = mean_a .* I + mean_b; % Eqn. (8) in the paper;
        Mat q = new Mat();
        Core.add(mean_a.mul(I), mean_b, q);
        q.convertTo(q, CvType.CV_32F);
        return q;
    }

    public static void showMyImage(Mat imBig, Mat im, int index){
        im.copyTo(imBig.submat(new Rect((index % 6) * im.cols(), (index / 6) * im.rows(), im.cols(), im.rows())));
        imshow("Ablak", imBig);
        index = (index + 1) % 18;
        waitKey();
    }

    // This is not working
    // 1. issue: does not increse the index
    // 2. issue: does not recognize the Z pic
    // 3. issue: need to find a solution for the negative arguments
    public static void lab02(){
        int index = 0;
        Mat im0 = imread("eper.jpg", 1);
        Mat imBig = new Mat(  im0.rows() * 3,  im0.cols() * 6, im0.type());
        imBig.setTo(new Scalar(128, 128, 255, 0));

        //Mat z = new Mat(im0.rows(), im0.cols(), CV_8UC1, new Scalar(0));

        // This should return a zero array of the specified size and type.
        // Maybe the GuidedImageFilter does not recognize the zero value
        Mat z = Mat.zeros(im0.rows(), im0.cols(), CV_8UC1);


        List<Mat> img = new ArrayList<>();
        Core.split(im0, img);
        int q = 8;
        double eps = 0.1 * 0.1;
        Mat r = GuidedImageFilter(img.get(0), img.get(0), q, eps);
        Mat g = GuidedImageFilter(img.get(1), img.get(1), q, eps);
        Mat b = GuidedImageFilter(img.get(2), img.get(2), q, eps);

        Mat result =  im0.clone();
        //original
        Core.merge(new ArrayList<>(Arrays.asList(r, g, b)), result);
        result.convertTo(result, CV_8UC1);
        showMyImage(imBig, result, index);
        /*
        // red is zero
        Core.merge(new ArrayList<>(Arrays.asList(z, g, b)), result);
        result.convertTo(result, CV_8UC1);
        showMyImage(imBig, result, index);
        // green is zero
        Core.merge(new ArrayList<>(Arrays.asList(r, z, b)), result);
        result.convertTo(result, CV_8UC1);
        showMyImage(imBig, result, index);
        // blue is zero
        Core.merge(new ArrayList<>(Arrays.asList(r, g, z)), result);
        result.convertTo(result, CV_8UC1);
        showMyImage(imBig, result, index);
        // green and blue are zero
        Core.merge(new ArrayList<>(Arrays.asList(r, z, z)), result);
        result.convertTo(result, CV_8UC1);
        showMyImage(imBig, result, index);
        // red and blue are zero
        Core.merge(new ArrayList<>(Arrays.asList(z, g, z)), result);
        result.convertTo(result, CV_8UC1);
        showMyImage(imBig, result, index);
        // red and green are zero
        Core.merge(new ArrayList<>(Arrays.asList(z, z, b)), result);
        result.convertTo(result, CV_8UC1);
        showMyImage(imBig, result, index);

         */
        //original
        Core.merge(new ArrayList<>(Arrays.asList(r, g, b)), result);
        result.convertTo(result, CV_8UC1);
        showMyImage(imBig, result, index);
        // 1. permutation
        Core.merge(new ArrayList<>(Arrays.asList(r, b, g)), result);
        result.convertTo(result, CV_8UC1);
        showMyImage(imBig, result, index);
        // 2. permutation
        Core.merge(new ArrayList<>(Arrays.asList(g, r, b)), result);
        result.convertTo(result, CV_8UC1);
        showMyImage(imBig, result, index);
        // 3. permutation
        Core.merge(new ArrayList<>(Arrays.asList(g, b, r)), result);
        result.convertTo(result, CV_8UC1);
        showMyImage(imBig, result, index);
        // 4. permutation
        Core.merge(new ArrayList<>(Arrays.asList(b, r, g)), result);
        result.convertTo(result, CV_8UC1);
        showMyImage(imBig, result, index);
        // 5. permutation
        Core.merge(new ArrayList<>(Arrays.asList(b, g, r)), result);
        result.convertTo(result, CV_8UC1);
        showMyImage(imBig, result, index);
        // red is negative
       /* Core.merge(new ArrayList<>(Arrays.asList(~r, g, b)), result);
        result.convertTo(result, CV_8UC1);
        showMyImage(imBig, result, index);
        // green is negative
        Core.merge(new ArrayList<>(Arrays.asList(r, ~g, b)), result);
        result.convertTo(result, CV_8UC1);
        showMyImage(imBig, result, index);
        // blue is negative
        Core.merge(new ArrayList<>(Arrays.asList(r, g, ~b)), result);
        result.convertTo(result, CV_8UC1);
        showMyImage(imBig, result, index);

        // ycrcb coding
        cvtColor(im0, result, COLOR_BGR2YCrCb);
        List<Mat> ycrcb = new ArrayList<>();
        Core.split(result, ycrcb);
        Mat y = GuidedImageFilter(img.get(0), img.get(0), q, eps);
        Mat cr = GuidedImageFilter(img.get(1), img.get(1), q, eps);
        Mat cb = GuidedImageFilter(img.get(2), img.get(2), q, eps);
        Core.merge(new ArrayList<>(Arrays.asList(~y, cr, cb)), result);
        cvtColor(im0, result, COLOR_YCrCb2BGR);
        result.convertTo(result, CV_8UC1);

        //originals negative
        Core.merge(new ArrayList<>(Arrays.asList(~r, ~g, ~b)), result);
        result.convertTo(result, CV_8UC1);
        showMyImage(imBig, result, index);
*/

    }

    public static void lab3_1(){
        Mat imBe = imread("mayans.jpg",1);
        Mat maszk = new Mat(3, 3, CV_32FC1);
        maszk.put(row,col,0.11, -0.08, 0.19, -0.13, 0.91, 0.31, -0.18, 0.02, -0.12);
        for (int i = 0; i < 100; ++i) {
            filter2D(imBe, imBe, -1, maszk);
            imshow("szurt kep", imBe);
            waitKey(0);
        }
    }

    public static void lab3_2(){
        Mat imBe = imread("mayans.jpg",1);
        Mat maszk = new Mat(3,3,CV_32FC1);
        maszk.put(row,col, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1);
        for( int i = 0 ; i < 20 ; ++i){
            filter2D(imBe,imBe,-1,maszk);
            imshow("szurt kep",imBe);
            waitKey(0);
        }
    }

    public static void lab3_3(){
        Mat imBe = imread("mayans.jpg",1);
        for( float k = 0; k < 20 ; ++k){
            Mat maszk = new Mat(3,3,CV_32FC1);
            maszk.put(row,col, 0, -k / 4, 0, -k / 4, 1 + k, -k / 4, 0, -k / 4, 0);
            filter2D(imBe,imBe,-1,maszk);
            imshow("szurt kep",imBe);
            waitKey(0);

        }

    }

    public static void lab3_4_Blur(){
        Mat imBe = imread("mayans.jpg",1);
        for(float k = 1; k < 21 ; k += 2){
            Imgproc.blur(imBe,imBe, new Size(k,k));
            imshow("szurt kep",imBe);
            waitKey(0);
        }
    }

    public static void lab3_4_GaussianBlur(){
        Mat imBe = imread("mayans.jpg",1);
        for(float k = 1; k < 21 ; k += 2){
            GaussianBlur(imBe,imBe,new Size(k,k),1);
            imshow("szurt kep",imBe);
            waitKey(0);
        }
    }

    public static void lab3_5_MedianBlur(){
        Mat imBe = imread("mayans.jpg", 1);
        double rand = Math.random();
        for (int db = 0; db < 20; ++db) {
            line(imBe,
                    new Point(rand % imBe.cols(), rand % imBe.rows()),
                    new Point(rand % imBe.cols(), rand % imBe.rows()),
                    new Scalar(0, 0, 0, 0),
                    1 + db % 2);
        }
        for (int i = 1; i < 20; i += 2) {
            Imgproc.medianBlur(imBe, imBe, i);
            imshow("Median Blur", imBe);
            waitKey(0);
        }
    }

    public static void lab3_6_MedianBLur(){
        Mat imBe = imread("amoba.png", 1);

        for (float k = 1; k < 1000; k +=2) {
            medianBlur(imBe, imBe, 21);
            imshow("szurt kep", imBe);
            waitKey(5);
        }
    }

    public static void lab4(){
        int index = 0;
        Mat imBe = imread("trondheim.jpg",1);
        Mat Mvp = new Mat(3,3,CV_32FC1);
        Mvp.put(row,col,
                -1, 0, 1,
                -1, 0, 1,
                -1, 0, 1);
        Mat imKi1 = imBe.clone();
        filter2D(imBe, imKi1, -1, Mvp);
        imshow("elso", imKi1);
        waitKey(0);
//----------------------------------------------------------
        Mat Mvn = new Mat(3,3,CV_32FC1);
        Mvn.put(row,col,
                1, 0, -1,
                1, 0, -1,
                1, 0, -1);
        Mat imKi2 = imBe.clone();
        filter2D(imBe, imKi2, -1, Mvn);
        imshow("masodik", imKi2);
        waitKey(0);
//--------------------------------------------------------------
        Mat Mfp = new Mat(3,3,CV_32FC1);
        Mfp.put(row,col,
                1, 1, 1,
                0, 0, 0,
                -1, -1, -1);
        Mat imKi3 = imBe.clone();
        filter2D(imBe, imKi3, -1, Mfp);
        imshow("harmadik", imKi3);
        waitKey(0);
    //--------------------------------------------------------------
        Mat Mfn = new Mat(3,3,CV_32FC1);
        Mfn.put(row,col,
                -1, -1, -1,
                0, 0, 0,
                1, 1, 1);
        Mat imKi4 = imBe.clone();
        filter2D(imBe, imKi4, -1, Mfn);
        imshow("negyedik", imKi4);
        waitKey(0);
        //--------------------------------------------------------------
        Mat M5 = new Mat(3,3,CV_32FC1);
        M5.put(row,col,
                0, -1, -1,
                1, 0, -1,
                1, 1, 0);
        Mat imKi5 = imBe.clone();
        filter2D(imBe, imKi5, -1, M5);
        imshow("otodik", imKi5);
        waitKey(0);
        //--------------------------------------------------------------
        Mat M6 = new Mat(3,3,CV_32FC1);
        M6.put(row,col,
                0, 1, 1,
                -1, 0, 1,
                -1, -1, 0);
        Mat imKi6 = imBe.clone();
        filter2D(imBe, imKi6, -1, M6);
        imshow("hatodik", imKi6);
        waitKey(0);
        //--------------------------------------------------------------
        Mat M7 = new Mat(3,3,CV_32FC1);
        M7.put(row,col,
                1, 1, 0,
                1, 0, -1,
                0, -1, -1);
        Mat imKi7 = imBe.clone();
        filter2D(imBe, imKi7, -1, M7);
        imshow("hetedik", imKi7);
        waitKey(0);
        //--------------------------------------------------------------
        Mat M8 = new Mat(3,3,CV_32FC1);
        M8.put(row,col,
                -1, -1, 0,
                -1, 0, 1,
                0, 1, 1);
        Mat imKi8 = imBe.clone();
        filter2D(imBe, imKi8, -1, M8);
        imshow("nyolcadik", imKi8);
        waitKey(0);
    }

    public static void lab4_Canny(){
        Mat imBe = imread("eper.jpg",1);
        imshow("eredeti",imBe);
        Mat gray = new Mat(imBe.rows(), imBe.cols(), imBe.type());
        Mat edges = new Mat(imBe.rows(), imBe.cols(), imBe.type());
        Mat dst = new Mat(imBe.rows(), imBe.cols(), imBe.type(), new Scalar(0));
        Imgproc.cvtColor(imBe, gray, Imgproc.COLOR_RGB2GRAY);
        Imgproc.Canny(gray,edges,50,150,3);
        edges.convertTo(dst,CV_8U);
        imshow("Canny",dst);
        waitKey(0);


    }

    public static void lab5_calcHist(){
        //String filename = args.length > 0 ? args[0] : "../data/lena.jpg";
        //Mat src = Imgcodecs.imread(filename);
        Mat src = imread("japan.jpg",1);
        if (src.empty()) {
            System.err.println("Cannot read image: " + "japan.jpg");
            System.exit(0);
        }
        List<Mat> bgrPlanes = new ArrayList<>();
        Core.split(src, bgrPlanes);
        int histSize = 256;
        float[] range = {0, 256}; //the upper boundary is exclusive
        MatOfFloat histRange = new MatOfFloat(range);
        boolean accumulate = false;
        Mat bHist = new Mat(), gHist = new Mat(), rHist = new Mat();
        Imgproc.calcHist(bgrPlanes, new MatOfInt(0), new Mat(), bHist, new MatOfInt(histSize), histRange, accumulate);
        Imgproc.calcHist(bgrPlanes, new MatOfInt(1), new Mat(), gHist, new MatOfInt(histSize), histRange, accumulate);
        Imgproc.calcHist(bgrPlanes, new MatOfInt(2), new Mat(), rHist, new MatOfInt(histSize), histRange, accumulate);
        int histW = 512, histH = 400;
        int binW = (int) Math.round((double) histW / histSize);
        Mat histImage = new Mat( histH, histW, CvType.CV_8UC3, new Scalar( 0,0,0) );
        Core.normalize(bHist, bHist, 0, histImage.rows(), Core.NORM_MINMAX);
        Core.normalize(gHist, gHist, 0, histImage.rows(), Core.NORM_MINMAX);
        Core.normalize(rHist, rHist, 0, histImage.rows(), Core.NORM_MINMAX);
        float[] bHistData = new float[(int) (bHist.total() * bHist.channels())];
        bHist.get(0, 0, bHistData);
        float[] gHistData = new float[(int) (gHist.total() * gHist.channels())];
        gHist.get(0, 0, gHistData);
        float[] rHistData = new float[(int) (rHist.total() * rHist.channels())];
        rHist.get(0, 0, rHistData);
        for( int i = 1; i < histSize; i++ ) {
            Imgproc.line(histImage, new Point(binW * (i - 1), histH - Math.round(bHistData[i - 1])),
                    new Point(binW * (i), histH - Math.round(bHistData[i])), new Scalar(255, 0, 0), 2);
            Imgproc.line(histImage, new Point(binW * (i - 1), histH - Math.round(gHistData[i - 1])),
                    new Point(binW * (i), histH - Math.round(gHistData[i])), new Scalar(0, 255, 0), 2);
            Imgproc.line(histImage, new Point(binW * (i - 1), histH - Math.round(rHistData[i - 1])),
                    new Point(binW * (i), histH - Math.round(rHistData[i])), new Scalar(0, 0, 255), 2);
        }
        HighGui.imshow( "Source image", src );
        HighGui.imshow( "calcHist Demo", histImage );
        HighGui.waitKey(0);

    }

}
