## AOP

Spring5 Demo 4

**AOP（概念）**

1、什么是AOP

（1）面向切面编程（面向方面编程），利用AOP可以对业务逻辑的各个部分进行隔离，从而使得各部分之间的耦合度降低，提高程序的可重用性，同时提高了开发的效率。

（2）通俗描述：不通过修改源代码方式，在主干功能里面添加 新功能

（3）使用登录例子说明AOP

![图3](D:\Markdown\images\图3.png)

## AOP（底层原理）

**1、AOP底层使用动态代理**

​	（1）有两种情况动态代理

​		第一种  有接口情况，使用JDK动态代理

​		创建接口实现类代理对象，增强类的方法

​		第二种  没有接口情况

![图4](D:\Markdown\images\图4.png)

## AOP（JDK动态代理）

1、使用JDK动态代理，使用Proxy类里面的方法创建代理对象

```java
java.lang.reflection
Class Proxy
```

（1）调用newProxyInstance方法

​	方法有三个参数：

```java
public static Object newProxyInstance(ClassLoader loader,
                                      Class<?>[] interfaces,
                                      InvocationHandler h)
```

​	第一参数：类加载器

​	第二参数：增强方法所在的类，这个类实现的接口，支持多个接口

​	第三参数：实现这个接口InvocationHandler，创建代理对象，写增强的部分

2、编写JDK动态代理代码

（1）创建接口，定义方法

```java
//（1）创建接口，定义方法
public interface UserDao {
 public int add(int a,int b);
 public String update(String id);
}
//（2）创建接口实现类，实现方法
public class UserDaoImpl implements UserDao {
 @Override
 public int add(int a, int b) {
 return a+b;
 }
 @Override
 public String update(String id) {
 return id;
 }
}
```

```java
//（3）使用 Proxy 类创建接口代理对象
public class JDKProxy {
 public static void main(String[] args) {
 //创建接口实现类代理对象
 Class[] interfaces = {UserDao.class};
 UserDaoImpl userDao = new UserDaoImpl(); 
/** 第一参数，类加载器 
	第二参数，增强方法所在的类，这个类实现的接口，(支持多个接口)
	第三参数，实现这个接口 InvocationHandler，创建代理对象，写增强的部分  */
 UserDao dao =(UserDao)Proxy.newProxyInstance(JDKProxy.class.getClassLoader(), interfaces,
					new UserDaoProxy(userDao));
 int result = dao.add(1, 2);
 System.out.println("result:"+result);
 }
}

//创建代理对象代码
class UserDaoProxy implements InvocationHandler {
 //1 把创建的是谁的代理对象，把谁传递过来
 //有参数构造传递
 private Object obj;
 public UserDaoProxy(Object obj) {
 this.obj = obj;
 }
 //增强的逻辑
 @Override
 public Object invoke(Object proxy, Method method, Object[] args) throws Throwable {
 //方法之前
 System.out.println("方法之前执行...."+method.getName()+" :传递的参数..."+ Arrays.toString(args));
 //被增强的方法执行
 Object res = method.invoke(obj, args);
 //方法之后
 System.out.println("方法之后执行...."+obj);
 return res;
 }
}
```

