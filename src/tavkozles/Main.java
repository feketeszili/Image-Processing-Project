package tavkozles;

import org.opencv.core.Core;

//4-6 legyen meg, 5-os nem muszaj max egy egyszeru hisztogram, es a 7-es nem kell
public class Main {
    static {System.loadLibrary(Core.NATIVE_LIBRARY_NAME);}
    public static void main(String[] args) {
	System.out.println(Core.VERSION);
	//-------------------------------------------------
	Functions.lab3_6_MedianBLur();

    }
}
