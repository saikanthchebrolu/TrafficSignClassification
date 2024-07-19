import java.rmi.*;
import java.rmi.server.*;

interface DBInterface extends Remote 
{
 public String input(String name1, String name2) throws RemoteException;
}
public class seever extends UnicastRemoteObject implements DBInterface 
{
 String name3;
 public seever() throws RemoteException 
 {
  try 
  {
   System.out.println("Initializing Server\nServer Ready");
  } catch (Exception e) 
  {
   System.out.println("ERROR: " + e.getMessage());
  }
 }
 
    /**
     *
     * @param args
     */
    public static void main(String[] args) 
 {
  try 
  {
   seever rs = new seever();
   java.rmi.registry.LocateRegistry.createRegistry(1007).rebind(
     "DBServ", rs);
  } 
  catch (RemoteException e) 
  {
   System.out.println("ERROR: " + e.getMessage());
  }
 }

    /**
     *
     * @param name1
     * @param name2
     * @return
     */
    @Override
 public String input(String name1, String name2) 
 {
  try 
  {
   name3 = name1.concat(name2);
  } 
  catch (Exception e) 
  {
   System.out.println("ERROR: " + e.getMessage());
  }
  return name3;
 }
}