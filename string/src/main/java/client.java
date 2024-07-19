import java.io.*;
import java.rmi.NotBoundException;
import java.rmi.registry.*;

public class client 
{
 static String name1, name2, name3;

 public static void main(String args[])
 {
   BufferedReader b = new BufferedReader(new
   InputStreamReader(System.in));
   int ch;
   try 
   {
     Registry r1 = LocateRegistry.getRegistry ( "localhost", 1007);
     DBInterface DI=(DBInterface)r1.lookup("DBServ");
     do
     {
       System.out.println("1.send input stings\n2.Display \n concatenated string \nEnter ur choice");
       ch= Integer.parseInt(b.readLine());
       switch(ch)
       {
          case 1:
          System.out.println(" \n Enter first string:");
          name1=b.readLine();
          System.out.println(" \n Enter second string:");
          name2=b.readLine();
          name3=DI.input(name1,name2);
          break;
          case 2:
          //display
          System.out.println("\nConcatenated String is : ");
          System.out.println(" "+name3 +"");
          ch=0;
          break;
       }
     }while(ch>0);
   }
   catch (IOException | NumberFormatException | NotBoundException e)
   {
     // System.out.println("ERROR: " +e.getMessage());
   }
 }
}