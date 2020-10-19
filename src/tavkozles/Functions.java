package tavkozles;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import java.lang.Math;
import java.util.ArrayList;
import static org.opencv.core.Core.addWeighted;
import static org.opencv.core.CvType.*;
import static org.opencv.highgui.HighGui.imshow;
import static org.opencv.highgui.HighGui.waitKey;
import static org.opencv.imgcodecs.Imgcodecs.imread;
import static org.opencv.imgcodecs.Imgcodecs.imwrite;
import static org.opencv.imgproc.Imgproc.*;

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

    public static void showMyImage(Mat imBig, Mat im, int index){
        im.copyTo(imBig.submat(new Rect((index % 6) * im.cols(), (index / 6) * im.rows(), im.cols(), im.rows())));
        imshow("Ablak", imBig);
        index = (index + 1) % 18;
        waitKey();
    }

    // this is not working
    public static void lab02(){
        int index = 0;
        Mat im0 = imread("eper.jpg", 1);
        Mat imBig = new Mat(  im0.rows() * 3,  im0.cols() * 6, im0.type());
        imBig.setTo(new Scalar(128, 128, 255, 0));

        Mat z = new Mat(im0.cols(), im0.rows(), CV_8UC1, new Scalar(0));


    }

    public static void lab5(){}

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

}
