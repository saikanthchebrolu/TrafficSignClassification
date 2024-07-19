
import java.net.*;
import java.io.*;
public class server {
    public static void main(String args[]){
        try{
            DatagramSocket s =  new DatagramSocket(1234);
            while ( true ) {
            DatagramPacket packet = new DatagramPacket(new byte[1024], 1024);
            s.receive( packet );
            String message = new String(packet.getData(), 0, 0, packet.getLength());
int res=1;
int ms=Integer.parseInt(message);
for(int i=1;i<=ms;i++) res=res*i;
String str1=res+" ";
  System.out.println( "Factorial of " + 
              message + " is " + str1);       }   }
  catch( IOException e){
    System.out.println(e);
}   
    }
}

    
   
