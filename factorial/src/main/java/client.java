import java.net.*;
import java.io.*;
public class client {
     public static void main(String ar[]) {
  int myPort = 1234;
try {

  DatagramSocket ds = new DatagramSocket();
  DatagramPacket pack;
  InetAddress addr = InetAddress.getLocalHost();
  BufferedReader b=new BufferedReader (new InputStreamReader(System.in));  
       {
         System.out.print("Enter the number to find factorial : ");
         String message=b.readLine();   
         byte [] data = new byte [ message.length() ];
         message.getBytes(0, data.length, data, 0);   
         pack =  new DatagramPacket(data, data.length, addr, myPort);
        ds.send( pack );
  }  }
        catch ( IOException e )  {
            System.out.println(e);
        }
   }
  }
