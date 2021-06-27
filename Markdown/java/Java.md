1、方法的定义格式
      修饰符 返回值类型 方法的名字 (参数列表...){
		   方法的功能主体
		     循环,判断,变量,比较,运算
		   return ;
	  }
	  

	 修饰符:  固定写法  public static
	 返回值类型:  方法在运算后,结果的数据类型
	 方法名:  自定义名字,满足标识符规范, 方法名字首字母小写,后面每个单词首字母大写
	 参数列表: 方法的运算过程中,是否有未知的数据, 如果有未知的数据,定义在参数列表上 (定义变量)
	 return: 方法的返回, 将计算的结果返回. 结束方法


2、方法定义和使用的注意事项
	 1. 方法不能定义在另一个方法的里面
	 2. 写错方法名字
	 3. 写错了参数列表
	 4. 方法返回值是void,方法中可以省略return 不写
	     return 下面不能有代码
	 5. 方法返回值类型,和return 后面数据类型必须匹配
	 6. 方法重复定义问题
	 7. 调用方法的时候,返回值是void, 不能写在输出语句中
	
3、方法的重载特性 (overload)
	在同一个类中,允许出现同名的方法,只要方法的参数列表不同即可,这样方法就是重载
	参数列表不同: 参数的个数,数据类型,顺序
4、方法,调用中的参数传递问题
     1. 方法参数是基本数据类型
	 2. 方法参数是引用类型
	    传递的是内存地址!!!

5、定义类:
	  使用类的形式,对现实中的事物进行描述
	  事物: 方法,属性
	    方法: 这个事物具备的功能
		属性: 变量

     格式:
       public class 类名{
    	    属性定义
    		  修饰符 数据类型 变量名 = 值
    		
    		方法定义
    		  修饰符 返回值类型  方法名(参数列表){
    			  
    		  }
       }
       
       要求: 使用Java中的类,描述生活中的手机这个事物
         手机事物,具备哪些属性, 属于他自己的特性
    	   颜色,品牌, 大小
1、ArrayList集合的使用
   也是*引用*数据类型
   步骤:
     1. 导入包 java.util包中
	 2. 创建引用类型的变量
	   数据类型< 集合存储的数据类型>  变量名 = new 数据类型 <集合存储的数据类型>  ();
	   集合存储的数据类型: 要将数据存储到集合的容器中
	   创建集合引用变量的时候,必须要指定好,存储的类型是什么
	   

	   ArrayList<String> array = new ArrayList<String>();
	   3. 变量名.方法 
	 
	 注意: 集合存储的数据,8个基本类型对应8个引用类型
	 存储引用类型,不存储基本类型
		**ArrayList<要存储元素的数据类型> 变量名 = new ArrayList<要存储元素的数据类型>();**
2、ArrayList 集合中的方法

   add(参数) 向集合中添加元素,数据存储进去
   方法中的参数类型,定义集合对象时候的类型是一致

   ArrayList<Integer> array = new ArrayList<Integer>();
   array.add(3);

   get(int index) 取出集合中的元素,get方法的参数,写入索引

   size() 返回集合的长度, 集合存储元素的个数
   集合没有length只有size
		//在索引2上,添加元素7
		array.add(2,7);
		
		//将0索引上的元素,修改成10
		array.set(0,10);
		
		//将4索引上的元素,删除
		array.remove(4);
		//清空集合中的元素，集合依然存在
		array.clear();
基本操作：
				增删改查



## 封装：
							封装表现：
							1、方法就是一个最基本封装体。
							2、类其实也是一个封装体。
							从以上两点得出结论，封装的好处：
							1、提高了代码的复用性。
							2、隐藏了实现细节，还要对外提供可以访问的方式。便于调用者的使用。这是核心之一，也可以理解为就是封装的概念。
							3、提高了安全性
this 关键字：
	就近访问原则，如果局部内有变量，则使用最近的变量
	当在方法中出现了局部变量和成员变量同名的时候，那么在方法中怎么区分局部变量和成员变量
	成员变量：类中定义的变量。局部变量：方法中定义的变量，
	二者中存在相同的名称时，可以利用this来指定类中的变量 
	*当构造方法中的参数与类中属性同名时，类中属性无法被正确赋值。
	这种情况下，可以使用this来指定类中成员，进而用来操作。*
继承：
class 子类 extends 父类｛｝
示例
				

```java
/*
				* 定义员工类Employee
				*/
				class Employee {
				String name; // 定义name属性
				// 定义员工的工作方法
				public void work() {
					System.out.println("尽心尽力地工作");
				}
				}
				
				/*
				* 定义研发部员工类Developer 继承 员工类Employee
				*/
				class Developer extends Employee {
				// 定义一个打印name的方法
				public void printName() {
					System.out.println("name=" + name);
				}
				}
				
				/*
				* 定义测试类
				*/
				public class Example01 {
				public static void main(String[] args) {
					Developer d = new Developer(); // 创建一个研发部员工类对象
					d.name = "小明"; // 为该员工类的name属性进行赋值
					d.printName(); // 调用该员工的printName()方法
					d.work(); // 调用Developer类继承来的work()方法
				}
				}
```
继承的好处：
***1、继承的出现提高了代码的复用性，提高软件开发效率。
2、继承的出现让类与类之间产生了关系，提供了多态的前提。***
在类的继承中，需要注意一些问题，具体如下：
	***1、在Java中，类只支持单继承，不允许多继承，也就是说一个类只能有一个直接父类，例如下面这种情况是不合法的。***
class A{} 
     class B{}
     class C extends A,B{}  // C类不可以同时继承A类和B类
	***2、多个类可以继承一个父类，例如下面这种情况是允许的。***
     class A{}
     class B extends A{}
     class C extends A{}   // 类B和类C都可以继承类A
	***3、在Java中，多层继承是可以的，即一个类的父类可以再去继承另外的父类，例如C类继承自B类，而B类又可以去继承A类，这时，C类也可称作A类的子类。下面这种情况是允许的。***
     class A{}
     class B extends A{}   // 类B继承类A，类B是类A的子类
     class C extends B{}   // 类C继承类B，类C是类B的子类，同时也是类A的子类
	*** ***
1.4	继承-子父类中成员变量的特点
了解了继承给我们带来的好处，提高了代码的复用性。继承让类与类或者说对象与对象之间产生了关系。那么，当继承出现后，类的成员之间产生了那些变化呢？
类的成员重点学习成员变量、成员方法的变化。
成员变量：如果子类父类中出现不同名的成员变量，这时的访问是没有任何问题。
看如下代码：
```java
class Fu
{
	//Fu中的成员变量。
	int num = 5;
}
class Zi extends Fu
{
	//Zi中的成员变量
	int num2 = 6;
	//Zi中的成员方法
	public void show()
	{
		//访问父类中的num
		System.out.println("Fu num="+num);
		//访问子类中的num2
		System.out.println("Zi num2="+num2);
	}
}
class Demo 
{
	public static void main(String[] args) 
	{
		Zi z = new Zi(); //创建子类对象
		z.show(); //调用子类中的show方法
	}
}

```

 *  继承后,子类父类中,成员变量的特点
 *  Zi extends Fu
 *  
 *  子类的对象,调用成员变量
 *    ***子类自己有,使用自己的***
 *    子类没有,调用父类的
 *    
 *   在子类中,调用父类的成员,关键字 super.调用父类的成员
 *   子类 (派生类)  继承父类  (超类,基类)
 *   
 *   this.调用自己本类成员
 *   super.调用的自己的父类成员
***当子父类中出现了同名成员变量时，在子类中若要访问父类中的成员变量，必须使用关键字super来完成。***
***this调用自己本类成员
super调用自己的父类成员***


***重写***
 *  继承后,子类父类中成员方法的特点
 *  
 *    子类的对象,调用方法的时候
 *      子类自己有,使用子类
 *      子类自己没有,调用的是父类
 *      
 ***重载: 方法名一样,参数列表不同,同一个类的事情
	 	重写=覆盖 : Override：子类中,出现了和父类一模一样的方法的时候, 子类重写父类的方法, 覆盖***


***抽象类的特点***
			1、抽象类和抽象方法都需要被abstract修饰。抽象方法一定要定义在抽象类中。
			2、抽象类不可以直接创建对象，原因：调用抽象方法没有意义。
			**3、只有覆盖了抽象类中所有的抽象方法后，其子类才可以创建对象。否则该子类还是一个抽象类。
			之所以继承抽象类，更多的是在思想，是面对共性类型操作会更简单。**
***抽象类一定是一个父类***		
			1、抽象类一定是个父类？	
					是的，因为不断抽取而来的。
			2、抽象类中是否可以不定义抽象方法。
			是可以的，那这个抽象类的存在到底有什么意义呢？不让该类创建对象,方法可以直接让子类去使用
			3、抽象关键字abstract不可以和哪些关键字共存？	
				1、private：私有的方法子类是无法继承到的，也不存在覆盖，而abstract和private一起使用修饰方法，abstract既要子类去实现这个方法，而private修饰子类根本无法得到

package cn.itcast.demo06;
/*
 *  定义类开发工程师类
 *    EE开发工程师 :  工作
 *    Android开发工程师 : 工作
 *    
 *    根据共性进行抽取,然后形成一个父类Develop
 *    定义方法,工作: 怎么工作,具体干什么呀
 *    
***抽象类,不能实例化对象, 不能new的
不能创建对象的原因:  如果真的让你new了, 对象.调用抽象方法,抽象方法没有主体,根本就不能运行
抽象类使用: 定义类继承抽象类,将抽象方法进行重写,创建子类的对象***

*public abstract class Develop {*
   //定义方法工作方法,但是怎么工作,说不清楚了,讲不明白
	//就不说, 方法没有主体的方法,必须使用关键字abstract修饰
	//抽象的方法,必须存在于抽象的类中,类也必须用abstract修饰
	public abstract void work();
}
抽象类的意思是：抽象类的子类都有这个功能，例如work，但你要调用这个work子类的时候，一定要进行覆盖，重写

*   抽象类,可以没有抽象方法,可以定义带有方法体的方法
 *   让子类继承后,可以直接使用
 * // private abstract void show();
     //抽象方法,需要子类重写, 如果父类方法是私有的,子类继承不了,也就没有了重写

## 接口
1、接口定义：
	与定义类的class不同，接口定义时需要使用interface关键字。
定义接口所在的仍为.java文件，虽然声明时使用的为interface关键字的编译后仍然会产生.class文件。这点可以让我们将接口看做是一种只包含了功能声明的特殊类。
定义格式：
public interface 接口名 {
抽象方法1;
抽象方法2;
抽象方法3;
}
使用interface代替了原来的class，其他步骤与定义类相同：
***	接口中的方法均为公共访问的抽象方法
	接口中无法定义普通的成员变量***
					/*
					 * 定义接口
					 *   使用关键字interface  接口名字
					 * 接口定义: 
					 *    成员方法,全抽象
					 *    不能定义带有方法体的方法
					 *    
					 * 定义抽象方法: **固定格式**
					 * 
					 *   **public abstract 返回值类型  方法名字(参数列表);
					 *   修饰符 public  写,或者不写,都是public**
					 *   
					 *  接口中成员变量的定义
					 *    成员变量的定义,具体要求
					 *    
					 *    要求 : **必须定义为常量**
					 *    固定格式:
					 *      public static final 数据类型 变量名 = 值
					 */
类与接口的关系为实现关系，即类实现接口。实现的动作类似继承，只是关键字不同，实现使用implements。

***接口中成员的特点***
 *    ***1. 成员变量的特点, 没有变量,都是常量***
 *    固定定义格式: public static final 数据类型 变量名 = 值
 ***public  权限
 static  可以被类名直接.调用
 final   最终,固定住变量的值***
 *    
// 
 * 注意: public static final 修饰符,在接口的定义中,可以省略不写
 *    但是,！！！不写不等于没有，即便是int x = 3 他仍然不可修改
 *    三个修饰符,还可以选择性书写
 *    
 *   2. 接口中的成员方法特点:
 *      public abstract 返回值类型 方法名(参数列表)
 *      修饰符  public abstract 可以不写,选择性书写
 *      但是,写不写,都有
 *      
 *   3. 实现类,实现接口,重写接口全部抽象方法,创建实现类对象
 *      实现类,重写了一部分抽象方法,实现类,还是一个抽象类
 */
   class + MyInterfaceImp1 + implements + MyInterface
   class               类                      实现               接口

定义类, 实现接口,重写接口中的抽象方法
创建实现类的对象
类实现接口, 可以理解为继承
关键字  implements
 *   class 类 implements 接口{
 *     重写接口中的抽象方法
 *   }
 *                        类                   实现                     接口
 *   class  MyInterfaceImpl    implements         MyInterface

***接口的多实现：***
多实现没有安全隐患，原因在于接口中的方法全是抽象，没有主体 

 *   类C,同时去实现2个接口,接口A,B
 *   作为实现类,C,***全部重写***两个接口的所有抽象方法,才能建立C类的对象
 *   
 ***   C类,在继承一个类的同时,可以实现多个接口***
	 	必须同时对继承的抽象类、接口中的抽象方法都要进行重写，才能够运行


## 接口之间的多继承 
day11: demo04

***一个接口可以继承多个接口***
/*
 *   实现接口C,重写C接口的全部抽象方法
 *   而且接口C,继承A,B
 *   D实现类,重写A,B,C三接口全部抽象方法
 *   
 *   问: Java中有多继承吗
 *    类没有多继承
 *    接口之间多继承
 */

## 接口和抽象类的区别
相同点:
	都位于继承的顶端,用于被其他类实现或继承;
	都不能直接实例化对象;
	都包含抽象方法,其子类都必须覆写这些抽象方法;
区别:
	抽象类为部分方法提供实现,避免子类重复实现这些方法,提高代码重用性;接口只能包含抽象方法;
	一个类只能继承一个直接父类(可能是抽象类),却可以实现多个接口;(接口弥补了Java的单继承)
	抽象类是这个事物中应该具备的你内容, 继承体系是一种 is..a关系
	接口是这个事物中的额外内容,继承体系是一种 like..a关系

二者的选用:
	优先选用接口,尽量少用抽象类;
	需要定义子类的行为,又要为子类提供共性功能时才选用抽象类;

## 多态性:day11demo5

[下面的文字抄的](https://bbs.csdn.net/topics/392556362?utm_medium=distribute.pc_relevant.none-task-discussion_topic-BlogCommendFromBaidu-1.control&depth_1-utm_source=distribute.pc_relevant.none-task-discussion_topic-BlogCommendFromBaidu-1.control)
多态性是对象多种表现形式的体现。

现实中，比如我们按下 F1 键这个动作：
如果当前在 Flash 界面下弹出的就是 AS 3 的帮助文档；
如果当前在 Word 下弹出的就是 Word 帮助；
在 Windows 下弹出的就是 Windows 帮助和支持。
同一个事件发生在不同的对象上会产生不同的结果。
多态的优点

1. 消除类型之间的耦合关系
2. 可替换性
3. 可扩充性
4. 接口性
5. 灵活性
6. 简化性
多态存在的三个必要条件

继承
重写
父类引用指向子类对象
比如：

Parent p = new Child();
当使用多态方式调用方法时，首先检查父类中是否有该方法，如果没有，则编译错误；如果有，再去调用子类的同名方法。

多态的好处：可以使程序有良好的扩展，并可以对所有类的对象进行通用处理。

以下是一个多态实例的演示，详细说明请看注释：

Test.java 文件代码：
public class Test {
    public static void main(String[] args) {
      show(new Cat());  // 以 Cat 对象调用 show 方法
      show(new Dog());  // 以 Dog 对象调用 show 方法
                
      Animal a = new Cat();  // 向上转型  
      a.eat();               // 调用的是 Cat 的 eat
      Cat c = (Cat)a;        // 向下转型  
      c.work();        // 调用的是 Cat 的 work
  }  
***从这之上是抄的***


 **Java实现多态有三个必要条件：继承、重写、向上转型。**
// 多态调用方法,方法必须运行子类的重写!!
		
		//Java中,对象的多态性,调用程序中的方法
		// 公式:  父类类型或者是接口类型   变量  = new 子类的对象();                                                                                                                                                                         
Fu f = new Zi();
package cn.itcast.demo05;

```java
public class Test {
	public static void main(String[] args) {
		// 多态调用方法,方法必须运行子类的重写!!
		
		//Java中,对象的多态性,调用程序中的方法
		// 公式:  父类类型或者是接口类型   变量  = new 子类的对象();
		Fu  f  = new Zi();
		f.show();
		
		//抽象类Animal,子类是Cat
		Animal a = new Cat();
		a.eat();
		
		//接口Smoking,实现类Student
		Smoking sk = new Student();
		sk.smoking();
	}
}
```

 *   多态中,成员特点
 *   Fu f = new Zi();
 * 	 f.a成员变量对应的是父类中的成员变量
 *   ***成员变量:*** 
 *     ***编译的时候, 参考父类中有没有这个变量,如果有,编译成功,没有编译失败***
 *     ***运行的时候, 运行的是父类中的变量值***
 *    ***编译运行全看父类***
 *     
 *   ***成员方法:***
 * f.show()需要在父类中有这个show方法，父类没有编译失败，如果子类有则运行子类的方法，如果子类没有这个show方法，则运行父类的方法，如果父类没有show方法，那么就报错了
 *     ***编译的时候, 参考父类中有没有这个方法,如果有,编译成功,没有编译失败***
 *     ***运行的时候, 运行的是子类的重写方法***
 *     
 *    ***编译看父类,运行看子类***
 * 注意：要看清是多态还是子类
 * 多态：	Fu a = new Zi();
 * 子类：	Zi  a = new Zi();这个是子类，完全可以在父类中没有show方法的条件下运行show(子类有)

关键字：Instanceof，
 *  运算符比较运算符, 结果真假值
 *  关键字, instanceof, 比较引用数据类型
 * 	 Person p = new Student();
 *   p  = new Teacher() //Person p = new Teacher()这种方法则重复定义错误
 *   上边先将p定义成学生，再将其定义为老师，最后p是老师类，经过instanceof判断是否为学生类时，则返回false            
 *  
 *   关键字 instanceof 比较, 一个引用类型的变量,是不是这个类型的对象
 *    p变量,是Student类型对象,还是Teacher类型对象
 *  
 *    引用变量 instanceof 类名
 *    p instanceof Student  比较,p是不是Student类型的对象,如果是,intanceof返回true
 *  
***多态中的转型***
Fu f = new Zi();
		向上转型：当有子类对象赋值给一个父类引用时，便是向上转型，***多态本身就是向上转型的过程。***
	父类类型  变量名 = new 子类类型();
	如：Person p = new Student();
		向下转型：一个已经向上转型的子类对象可以使用强制类型转换的格式，将父类引用转为子类引用，这个过程是向下转型。如果是直接创建父类对象，是无法向下转型的！
	子类类型 变量名 = (子类类型) 父类类型的变量;
	如:Student stu = (Student) p;  //变量p 实际上指向Student对象
	package cn.itcast.demo08;
/*
 *  测试类
 *    1. 实现动物和Cat,Dog多态调用
 *    2. 做类型的强制转换,调用子类的特有功能
 */

```java
public class Test {
	public static void main(String[] args) {
		//两个子类,使用两次多态调用
		Animal a1 = new Cat();
		Animal a2 = new Dog();
		//a1,a2调用子类父类共有方法,运行走子类的重写
		a1.eat();
		a2.eat();
		
		//类型向下转型,强制转换,调用子类的特有
		//防止发生异常: a1属于Cat对象,转成Cat类,  a2属于Dog对象,转成Dog
		//instanceof判断
		
		if(a1 instanceof Cat){
			Cat c = (Cat)a1;
			c.catchMouse();
		}
		if(a2 instanceof Dog){
			Dog d = (Dog)a2;
			d.lookHome();
		}
```
	}

}

## 构造方法：
示例：***Person p = new Person("张三",20);***	
定义类必须拥有构造方法，构造方法不写也有
与类同名的方法，可以在定义对象时进行参数传递
	自定义的Person类，成员变量，name，age
	要求在new Person的同时，就制定好name，age的值
	实现功能，利用方法去实现，构造方法，构造器Constructor
	作用：在new的同时对成员变量赋值，给对象的属性初始化赋值new Person 对属性name，age赋值
 *  构造方法的定义格式
 *    权限  方法名(参数列表){
 *    }
 *    方法的名字,必须和类的名字完全一致
 *    构造方法不允许写返回值类型  , void 也不能写
 *    
 *    构造方法在什么时候,运行呢, 在new 的时候,自动执行
 *    只运行一次,仅此而已
 *    
 *    每个class必须拥有构造方法,构造方法不写也有
 *    编译的时候,javac, 会自动检查类中是否有构造方法
 *    如果有,就这样的
 *    如果没有,编译器就会自动添加一个构造方法
 *      编译器自动添加的构造方法: public Person(){}
 *    自己手写了构造方法,编译的时候,不会自动添加构造方法!
		构造方法的细节：
1、***一个类中可以有多个构造方法，多个构造方法是以重载的形式存在的***
2、构造方法是可以被private修饰的，作用：其他程序无法创建该类的对象。
## this关键字在构造方法之间调用

```java
package cn.itcast.demo03;
/*
 *   this可以在构造方法之间进行调用
 *   this.的方式,区分局部变量和成员变量同名情况
 *   this在构造方法之间的调用,语法 this()
 */
public class Person {
	private String name;
	private int age;
	
	public Person(){
		//调用了有参数的构造方法
		//参数李四,20传递给了变量name,age
		this("李四",20);
	}
	/*
	 *  构造方法,传递String,int
	 *  在创建对象的同时为成员变量赋值
	 */
	public Person(String name,int age){
		this.name = name;
		this.age = age;
	}
	
	public String getName() {
		return name;
	}
	public void setName(String name) {
		this.name = name;
	}
	public int getAge() {
		return age;
	}
	public void setAge(int age) {
		this.age = age;
	}
	
	
}
```
## super关键字
在创建子类对象时，父类的构造方法会先执行，因为子类中所有构造方法的第一行有默认的隐式super();语句
格式：				
	调用本类中的构造方法
	this(实参列表);
	调用父类中的空参数构造方法
	super();
	调用父类中的有参数构造方法
		super(实参列表);

***子类中,super()的方式,调用父类的构造方法***day12 demo04
 *  super()调用的是父类的空参数构造
 *  super(参数) 调用的是父类的有参数构造方法
 *  
 *  子类的构造方法, 有一个默认添加的构造方法
 *  注意: 子类构造方法的第一行,有一个隐式代码 super()
 *  		 public Student(){
 *      		 super();
 *   		 }
 *   子类的构造方法第一行super语句,调用父类的构造方法
 */

 ***子类构造方法的报错原因:找不到父类的空参数构造器***
 *  子类中,没有手写构造,编译器添加默认的空参数
 *  public Student(){
 *     super();
 *  }
 *  编译成功,必须手动编写构造方法,请你在super中加入参数
 *  
 *  注意: 子类中所有的构造方法,无论重载多少个,第一行必须是super()
 *  如果父类有多个构造方法,子类任意调用一个就可以
 *  super()语句必须是构造方法第一行代码
    	构造方法第一行,写this()还是super()
	 	不能同时存在,任选其一,
	 	***保证子类的所有构造方法调用到父类的构造方法即可***

 	***小结论: 无论如何,子类的所有构造方法,直接,间接必须调用到父类构造方法***
 	子类的构造方法,什么都不写,默认的构造方法第一行 super();
package cn.itcast.demo07;
/*
 *  Student类和Worker有相同成员变量,name age
 *  继承的思想,共性抽取,形成父类
 *  Person,抽取出来父类
 *  成员变量,私有修饰
 *  同时需要在创建学生和工人对象就必须明确姓名和年龄
 *  new Student, new Worker 姓名,年龄明确了
 */

```java
public class Person {
	private String name;
	private int age;
	
	public Person(String name,int age){
		this.name = name;
		this.age = age;
	}
	
	public String getName() {
		return name;
	}
	public void setName(String name) {
		this.name = name;
	}
	public int getAge() {
		return age;
	}
	public void setAge(int age) {
		this.age = age;
	}
	
}
代码2
package cn.itcast.demo07;

public class Student extends Person{
	public Student(String name,int age){
		super(name,age);
	}
}

代码3
package cn.itcast.demo07;

public class Test {
	public static void main(String[] args) {
		//创建工人对象, Worker,指定姓名年龄
		Worker w = new Worker("光头强", 30);
		System.out.println(w.getName());
		System.out.println(w.getAge());
		
		//创建学生对象, Student,指定姓名年龄
		Student s = new Student("肥波", 5);
		System.out.println(s.getName());
		System.out.println(s.getAge());
	}
}

```

## final关键字

***final的概念：***
		继承的出现提高了代码的复用性，并方便开发。但随之也有问题，有些类在描述完之后，不想被继承，或者有些类中的部分方法功能是固定的，不想让子类重写。可是当子类继承了这些特殊类之后，就可以对其中的方法进行重写，那怎么解决呢？
		要解决上述的这些问题，需要使用到一个关键字final，final的意思为最终，不可变。final是个修饰符，它可以用来修饰类，类的成员，以及局部变量。
	在类的定义上,加上修饰符,final
		 *  **类: 最终类, 不能有子类,不可以被继承**
		 *  **但是使用方式,没有变化,创建对象,调用方法**

final的特点：
1、final修饰类不可以被继承，但是可以继承其他类。
```java
class Yy {}
	final class Fu extends Yy{} //可以继承Yy类
	class Zi extends Fu{} //不能继承Fu类
```

2、final修饰的方法不可以被覆盖,但父类中没有被final修饰方法，子类覆盖后可以加final。
```java
class Fu {
	// final修饰的方法，不可以被覆盖，但可以继承使用
    public final void method1(){}
    public void method2(){}
}
class Zi extends Fu {
	//重写method2方法
	public final void method2(){}
}
```
3、final修饰的变量称为常量，这些变量只能赋值一次
```java
final int i = 20;
i = 30; //赋值报错，final修饰的变量只能赋值一次
```
4、引用类型的变量值为对象地址值，地址值不能更改，但是地址内的对象属性值可以修改。
```java
final Person p = new Person();
Person p2 = new Person();
p = p2; //final修饰的变量p，所记录的地址值不能改变
p.name = "小明";//可以更改p对象中name属性值
```
5、修饰成员变量，需要在创建对象前赋值，否则报错。(当没有显式赋值时，多个构造方法的均需要为其赋值。)
```java
class Demo {
	//直接赋值
	final int m = 100;
	
	//final修饰的成员变量，需要在创建对象前赋值，否则报错。
	final int n; 
	public Demo(){
		//可以在创建对象时所调用的构造方法中，为变量n赋值
		n = 2016;
	}
}
```

```java
package cn.itast.demo02;

public class Test {
	public static void main(String[] args) {
		Zi z = new Zi();
		z.function();
		z.show();
		
		final int i = 10;//被final修饰,一次赋值,终身不变
		System.out.println(i);
		
		//final修饰引用变量问题
		//变量,保存内存地址,终身不变  
		final Zi z2 = new Zi();
		z2.function();
		
	}
}
```
无论final修饰基本类型还是引用类型，一旦final后，终身不能改变。

final修饰成员变量
成员变量，在堆内存，具有默认值
final修饰的成员变量，固定的不是内存的默认值
固定的是，成员变量的手动赋值，绝对不是内存的默认
```java
final int age;//报错
final int age = 10;//正确写法
```

成员变量的赋值，2种实现方式，
**法一：是定义的时候，直接=赋值。
   法二：是赋值方式，采用构造方法赋值**    
	保证：被final修饰的成员变量，只能被赋值一次
	final int n;
		n = 100;  //这样也是错误，**如果一开始没有定义初值，那么只能通过方法赋值。setAge方法也不行
		，因为setAge是在建立对象之后的操作，final必须在建立对象时进行定义**
	成员变量，需要在创建对象前赋值，否则报错
	构造方法，是创建对象中的事情，可以为成员

## static关键字
1、static概念：
		当在定义类的时候，类中都会有相应的属性和方法。而属性和方法都是通过创建本类对象调用的。当在调用对象的某个方法时，这个方法没有访问到对象的特有数据时，方法创建这个对象有些多余。可是不创建对象，方法又调用不了，这时就会想，那么我们能不能不创建对象，就可以调用方法呢？
		可以的，我们可以通过static关键字来实现。static它是静态修饰符，一般用来修饰类中的成员。
2、static特点：
**（1）被static修饰的成员变量属于类，不属于这个类的某个对象。**（也就是说，多个对象在访问或修改static修饰的成员变量时，其中一个对象将static成员变量值进行了修改，其他对象中的static成员变量值跟着改变，即多个对象共享同一个static成员变量）

```java
public class Test {
	public static void main(String[] args) {
		System.out.println(Person.className);
		Person p1 = new Person();
		Person p2 = new Person();
		
		p1.name = "哈哈";
		p2.name = "嘻嘻";
		System.out.println(p1.name);
		System.out.println(p2.name);
		
		//对象调用类的静态成员变量
		p1.className = "基础班";
		System.out.println(p2.className);
		//className是静态的，定义一次之后，无论谁调用都是一个东西
	}
}
```

**（2）被static修饰的成员*可以并且建议*通过类名直接访问。**通过类名访问
	**访问静态成员的格式：
	类名.静态成员变量名
	类名.静态成员方法名(参数)**
	对象名.静态成员变量名     	------不建议使用该方式，会出现警告
	对象名.静态成员方法名(参数) 	------不建议使用该方式，会出现警告
2.3	static注意事项
（1）静态内容是优先于对象存在，只能访问静态，不能使用this/super。静态修饰的内容存于静态区。
（2）同一个类中，静态成员只能访问静态成员
（3）main方法为静态方法仅仅为程序执行入口，它不属于任何一个对象，可以定义在任意类中。
2.4	定义静态常量
	开发中，我们想在类中定义一个静态常量，通常使用public static final修饰的变量来完成定义。此时变量名用全部大写，多个单词使用下划线连接。
定义格式：
	public static final 数据类型 变量名 = 值;

方法里面如果都是静态变量，方法也应该加静态
如果方法里面存在静态和非静态，那么方法不应该加静态


静态的注意事项
**在静态中不能调用非静态**
为什么呢? 为什么静态不能调用非静态,生命周期
**静态优先于非静态存在于内存中**
静态 前人 先人   非静态 后人

**静态不能写this,不能写super**
问题:  static 修饰到底什么时候使用,应用场景
static 修饰成员变量,成员方法
成员变量加static, 根据具体事物,具体分析问题
定义事物的时候,多个事物之间是否有共性的数据!!
请你将共性的数据定义为静态的成员变量

**成员方法加static, 跟着变量走
如果方法,没有调用过非静态成员,将方法定义为静态**
每调用过非静态成员，则都定义为静态

多态调用中,编译看谁,运行看谁
编译都看 = 左边的父类, 父类有编译成功,父类没有编译失败
	 ***运行,静态方法, 运行父类中的静态方法（先执行父类，所以父类先定义了静态方法）
	 运行,非静态方法,运行子类的重写方法***
	 成员变量,编译运行全是父类
```java
public class Test {

	public static final double PI = 3.14159265358979323846;
	
	public static void main(String[] args) {
		Fu f = new Zi();   //
//		System.out.println(f.i);
		//调用还是父类的静态方法,原因: 静态属于类,不属于对象     
		//对象的多态性,静态和对象无关,父类的引用.静态方法
		f.show();
		System.out.println();
	}
}
```

例子
```java
package cn.itast.demo03;
/*
 *   定义Person类,
 *   定义对象的特有数据,和对象的共享数据
 *   对象的特有数据(非静态修饰) 调用者只能是 new 对象
 *   对象的共享数据(静态修饰)  调用者可以是new 对象,可以是类名
 *   
 *   被静态修饰的成员,可以被类名字直接调用
 */
public class Person {
	String name;
	
	static String className;
}
package cn.itast.demo03;

public class Student {

	private static String name;
	private static  int age ;
	private char sex;
	 
	 public static void function(){
		 System.out.println(name+age);
	 }
	 
	 public static int getSum(int a,int b,int c){
		 return a+b+c;
	 }
	 
	 public void show2(){
		 System.out.println(sex);
	 }
	 
	 public void show(){
		 System.out.println(name+age);
	 }
	 
	 public static void main(String[] args) {

	}
	 
	 public static void method(){
		 
	 }
	 
}
package cn.itast.demo03;

public class Test {
	public static void main(String[] args) {
		System.out.println(Person.className);
		Person p1 = new Person();
		Person p2 = new Person();
		
		p1.name = "哈哈";
		p2.name = "嘻嘻";
		System.out.println(p1.name);
		System.out.println(p2.name);
		
		//对象调用类的静态成员变量
		p1.className = "基础班";
		System.out.println(p2.className);
		//className是静态的，定义一次之后，无论谁调用都是一个东西
	}
}

```

4.1	内部类概念
（1）什么是内部类
将类写在其他类的内部，可以写在其他类的成员位置和局部位置，这时写在其他类内部的类就称为内部类。其他类也称为外部类。
（2）什么时候使用内部类
在描述事物时，若一个事物内部还包含其他可能包含的事物，比如在描述汽车时，***汽车***中还包含这发动机，这时***发动机***就可以使用***内部类***来描述。

 *   内部类的定义
 *     将内部类,定义在了外部的成员位置
 *   类名Outer,内部类名Inner
 *   
 *   成员内部类,可以使用成员修饰符,public static ....
 *   也是个类,可以继承,可以实现接口
 *   
 *   调用规则: 内部类,可以使用外部类成员,包括私有
 *   外部类要使用内部类的成员,必须建立内部类对象
```java
public class Outer {
	private int a = 1;
	//外部类成员位置,定义内部类
    class Inner{
		public void inner(){
			System.out.println("内部类方法inner "+a);
		}
	}
}
```

```java
class 汽车 { //外部类
	class 发动机 { //内部类
}
}
```
（3）内部类的分类
**内部类**分为**成员内部类**与**局部内部类**。
我们定义内部类时，就是一个正常定义类的过程，同样包含各种修饰符、继承与实现关系等。**在内部类中可以直接访问外部类的所有成员。**
**4.2	成员内部类**  在成员变量位置的内部类
***成员内部类，定义在外部类中的成员位置***。与类中的成员变量相似，*可通过外部类对象进行访问*
（1）定义格式
class 外部类 { 
	修饰符 class 内部类 {
		//其他代码
}
}
（2）访问方式
外部类名.内部类名 变量名 = new 外部类名().new 内部类名();

（3）成员内部类代码演示
定义类

```java
class Body {//外部类，身体
	private boolean life= true; //生命状态
    public class Heart { //内部类，心脏
		public void jump() {
        	 System.out.println("心脏噗通噗通的跳")
			 System.out.println("生命状态" + life); //访问外部类成员变量
			}
		}
	}

```
访问内部类

```java
public static void main(String[] args) {
	//创建内部类对象
	Body.Heart bh = new Body().new Heart();
	//调用内部类中的方法
	bh.jump();
}
```
**4.3	局部内部类**：在方法中的内部类
***局部内部类，定义在外部类方法中的局部位置***。***与访问方法中的局部变量相似，可通过调用方法进行访问***
（1）定义格式
			class 外部类 { 
					修饰符 返回值类型 方法名(参数) {
							class 内部类 {
							//其他代码
							}
					}
			}

（2）局部内部类代码演示
	定义类
		
*好的例子：*重要
```java
class Party {//外部类，聚会
				public void puffBall(){// 吹气球方法
						class Ball {// 内部类，气球
           						public void puff(){
     									System.out.println("气球膨胀了");
								}
						}
						//创建内部类对象，调用puff方法
						new Ball().puff();
				}
		}
```
	访问内部类
		public static void main(String[] args) {
			//创建外部类对象
				Party p = new Party();
				//调用外部类中的puffBall方法
				p.puffBall();
		}

4.4 内部类的实际使用--匿名内部类
1.匿名内部类概念
	最常用到的内部类就是匿名内部类，它是局部内部类的一种。
定义的匿名内部类有两个含义：
			（1）临时定义某一指定类型的子类
			（2）定义后即刻创建刚刚定义的这个子类的对象

2.定义匿名内部类的作用与格式
	作用：匿名内部类是创建某个类型子类对象的快捷方式。
	格式：
```java
new 父类或接口(){
	//进行方法重写
};
```

```java
//已经存在的父类：
public abstract class Person{
	public abstract void eat();
}
//定义并创建该父类的子类对象，并用多态的方式赋值给父类引用变量
Person  p = new Person(){
	public void eat() {
		System.out.println(“我吃了”);
}
};
//调用eat方法
p.eat();
```
使用匿名对象的方式，将定义子类与创建子类对象两个步骤由一个格式一次完成，。虽然是两个步骤，但是两个步骤是连在一起完成的。
匿名内部类如果不定义变量引用，则也是匿名对象。代码如下：

```java
new Person(){
	public void eat() {
		System.out.println(“我吃了”);
}
}.eat();
```

```java
/*
 *  实现类,实现接口 重写接口抽象方法,创建实现类对象
 *  class XXX implements Smoking{
 *      public void smoking(){
 *      
 *      }
 *  }
 *  XXX x = new XXX();
 *  x.smoking(); 
 *  Smoking s = new XXX();
 *  s.smoking();
 *  
 *  匿名内部类,简化问题:  定义实现类,重写方法,建立实现类对象,合为一步完成
 */
new Smoking(){
			public void smoking(){
				System.out.println("人在吸烟");
			}
		}.smoking();
注释中的实现类和匿名内部类实现的功能一致
```
package cn.itast.demo09;

```java
public class Test {
	public static void main(String[] args) {
		//使用匿名内部类
		/*
		 *  定义实现类,重写方法,创建实现类对象,一步搞定
		 *  格式:
		 *    new 接口或者父类(){
		 *       重写抽象方法
		 *    };
		 *    从 new开始,到分号结束
		 *    创建了接口的实现类的对象
		 */
		new Smoking(){
			public void smoking(){
				System.out.println("人在吸烟");
			}
		}.smoking();
	}
}
```

## 5、包的声明与访问
5.1 包的概念
		java的包，其实就是我们电脑系统中的文件夹，包里存放的是类文件。
		当类文件很多的时候，通常我们会采用多个包进行存放管理他们，这种方式称为分包管理。
		在项目中，我们将相同功能的类放到一个包中，方便管理。并且日常项目的分工也是以包作为边界。
		类中声明的包必须与实际class文件所在的文件夹情况相一致，即类声明在a包下，则生成的.class文件必须在a文件夹下，否则，程序运行时会找不到类。
5.2 包的声明格式
		通常使用公司网址反写，可以有多层包，包名采用全部小写字母，多层包之间用”.”连接
	类中包的声明格式： 
```java
package 包名.包名.包名…;
```
	如：黑马程序员网址itheima.com那么网址反写就为com.itheima
	    传智播客 itcast.cn  那么网址反写就为 cn.itcast
	注意：声明包的语句，必须写在程序有效代码的第一行（注释不算）
	代码演示：
```java
package cn.itcast; //包的声明，必须在有效代码的第一行

import java.util.Scanner;
import java.util.Random;

public class Demo {}
```
5.3	包的访问
在访问类时，为了能够找到该类，必须使用含有包名的类全名（包名.类名）。

包名.包名….类名
如： java.util.Scanner
     java.util.Random
	cn.itcast.Demo
***带有包的类，创建对象格式：包名.类名 变量名 = new包名.类名();***
     cn.itcast.Demo d = new cn.itcast.Demo();

1、前提：包的访问与访问权限密切相关，这里以一般情况来说，即类用public修饰的情况。

2、类的简化访问
		**当我们要使用一个类时，这个类与当前程序在同一个包中（即同一个文件夹中），或者这个类是java.lang包中的类时通常可以省略掉包名，直接使用该类。**
		如：cn.itcast包中有两个类，PersonTest类，与Person类。我们在PersonTest类中，访问Person类时，由于是同一个包下，访问时可以省略包名，即直接通过类名访问 Person。
5.4	import导包
我们每次使用类时，都需要写很长的包名。很麻烦，我们可以通过import导包的方式来简化。
可以通过导包的方式使用该类，可以避免使用全类名编写（即，包类.类名）。
导包的格式：
import 包名.类名;

	当程序导入指定的包后，使用类时，就可以简化了。演示如下
```java
//导入包前的方式
//创建对象
java.util.Random r1 = new java.util.Random();
java.util.Random r2 = new java.util.Random();
java.util.Scanner sc1 = new java.util.Scanner(System.in);
java.util.Scanner sc2 = new java.util.Scanner(System.in);

//导入包后的方式
import java.util.Random;
import java.util.Scanner;
import java.util.*
//创建对象
Random r1 = new Random();
Random r2 = new Random();
Scanner sc1 = new Scanner(System.in);
Scanner sc2 = new Scanner(System.in);

```
	import导包代码书写的位置：在声明包package后，定义所有类class前，使用导包import包名.包名.类名;

## 访问修饰符

在Java中提供了四种访问权限，使用不同的访问权限时，被修饰的内容会有不同的访问权限，以下表来说明不同权限的访问能力：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201126202628348.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQwMzEwNzEw,size_16,color_FFFFFF,t_70#pic_center)
权限：
default：本包和自己能用，子类不能用
**private**：本类中访问（常用）
protected：本包中与其他包中的子类可以访问
**public**：所有包中的所有类都可以访问使用public（常用）

## 7、代码块
7.1	局部代码块
局部代码块是定义在方法或语句中
特点：
	1、以”{}”划定的代码区域，此时只需要关注作用域的不同即可
	2、方法和类都是以代码块的方式划定边界的
```java
class Demo{
	public static void main(String[] args)	{
		{
         		 int x = 1;
        		 System.out.println("普通代码块" + x);
		}
		int x = 99;
		System.out.println("代码块之外" + x);
	}
}
```
7.2	构造代码块
构造代码块是定义在类中成员位置的代码块
特点：
	1、**优先于构造方法执行**，构造代码块用于执行所有对象均需要的初始化动作
	2、每创建一个对象均会执行一次构造代码块。
```java
public class Person {
	private String name;
	private int age;
	
     //构造代码块
	{
		System.out.println("构造代码块执行了");
	}
	Person(){
		System.out.println("Person无参数的构造函数执行");
	}
	Person(int age){
		this.age = age;
		System.out.println("Person（age）参数的构造函数执行");
	}
}
class PersonDemo{
	public static void main(String[] args)	{
		Person p = new Person();
		Person p1 = new Person(23);
	}
}
```

7.3	静态代码块
静态代码块是定义在成员位置，使用static修饰的代码块。
特点：
	1、它优先于主方法执行、优先于构造代码块执行，当以任意形式第一次使用到该类时执行。
	2、该类不管创建多少对象，静态代码块只执行一次。
	3、可用于给静态变量赋值，用来给类进行初始化。
```java
package cn.itast.demo12;
/*
 *  静态代码块, 只执行一次
 *  构造代码块,new一次,就执行一次,优先于构造方法
 *  构造方法, new 一次,就执行一次
 */
public class Person {
  private String name;
  private int age;
  

  public Person(String name,int age){
	  this.age = age;
	  this.name = name;
	  System.out.println("我是构造方法");
  }
  //构造代码块
  {
	  System.out.println("我是构造代码块");
  }
  
  //静态代码块
  static{
	  System.out.println("我是静态代码块");
  }
}
//Test
package cn.itast.demo12;

public class Test {
	public static void main(String[] args) {
		new Person("张三",20);
		new Person("张三2",220);
	}
}

```
结果:
			我是静态代码块
			我是构造代码块
			我是构造方法
			我是构造代码块
			我是构造方法
 *  静态代码块, 只执行一次
 *  构造代码块,new一次,就执行一次,优先于构造方法
 *  构造方法, new 一次,就执行一次

## 第2章 面向对象

2.1	不同修饰符使用细节
常用来修饰类、方法、变量的修饰符如下：
	public 权限修饰符，公共访问, 类,方法,成员变量
	protected 权限修饰符，受保护访问, 方法,成员变量
	默认什么也不写 也是一种权限修饰符，默认访问, 类,方法,成员变量
	private 权限修饰符，私有访问, 方法,成员变量
	static 静态修饰符  方法,成员变量
	final 最终修饰符   类,方法,成员变量,局部变量
	abstract 抽象修饰符  类 ,方法

我们编写程序时，权限修饰符一般放于所有修饰符之前，不同的权限修饰符不能同时使用；
**同时，abstract与private不能同时使用；*abstract类必须重写，而private、static、final存在限制*
同时，abstract与static不能同时使用；
同时，abstract与final不能同时使用。**

public static void main(String[] args) {
		//调用方法operatorPerson,传递Person类型对象
		Person p = new Person();
		operatorPerson(p);
	
		operatorPerson(new Person());
	}

修饰类能够使用的修饰符：
**修饰类只能使用public、默认的、final、abstract关键字
使用最多的是 public关键字**
		public class Demo {} **//最常用的方式**
		class Demo2{}
		public final class Demo3{}
		public abstract class Demo4{}

	修饰**成员变量**能够使用的修饰符：
		public : 公共的
		protected : 受保护的
			: 默认的
		private ：私有的
		final : 最终的
		static : 静态的
**修饰成员变量使用最多的是 private**
		public int count = 100;
		protected int count2 = 100;
		int count3 = 100;
		private int count4 = 100; **//最常用的方式**
		public final int count5 = 100;
		public static int count6 = 100;
	**修饰构造方法**能够使用的修饰符：
		public : 公共的
		protected : 受保护的
			: 默认的
		private ：私有的
		使用最多的是 public
		public Demo(){} //最常用的方式
		protected Demo(){}
		Demo(){}
		private Demo(){}
	修饰成员方法能够使用的修饰符：
		public : 公共的
		protected : 受保护的
			: 默认的
		private ：私有的
		final : 最终的
		static : 静态的
		abstract : 抽象的
		使用最多的是 public
```java
public void method1(){}//最常用的方式
protected void method2(){}
void method3(){}
private void method4(){}
public final void method5(){}
public static void method6(){}//最常用的方式
public abstract void method7();//最常用的方式
```

## 自定义数据类型的使用
**3.1	辨析成员变量与方法参数的设计定义**
	定义长方形类，包含求周长与求面积的方法
	定义数学工具类，包含求两个数和的二倍与求两个数积的方法
思考：这两个类的计算方法均需要两个数参与计算，请问两个数定义在成员位置还是形参位置更好，为什么？
如果变量是该类的一部分时，定义成成员变量。 
如果变量不应该是类的一部分，而仅仅是功能当中需要参与计算的数，则定义为形参变量。

	数学工具类

```java
public class MathTool {
	//求两个数的和的二倍
	public double sum2times(int number,int number2) {
		return (number+number2)*2;
	}
	//求两个数的积
	public double area(int number,int number2) {
		return number*number2;
	}
}
```
	长方形类

```java
public class CFX {
	//因为长与宽，在现实事物中属于事物的一部分，所以定义成员变量
	private int chang;
	private int kuan;
	
	public CFX(int chang, int kuan) {
		this.chang = chang;
		this.kuan = kuan;
	}
	//求长与宽的周长
	public double zhouChang() {
		return (chang+kuan)*2;
	}
	//求长与宽的面积
	public double mianJi() {
		return chang*kuan;
	}
	public int getChang() {
		return chang;
	}
	public void setChang(int chang) {
		this.chang = chang;
	}
	public int getKuan() {
		return kuan;
	}
	public void setKuan(int kuan) {
		this.kuan = kuan;
	}
}

```
**3.2	类作为方法参数与返回值**
Person类当做方法的参数
Person类型写在方法的参数列表中
```java
package cn.itcast.classes;

public class Person {
	private String name = "张三";
	
	public void eat(){
		System.out.println(name+ "  在吃饭");
	} 
	
	public void run(){
		System.out.println(name+" 在跑步");
	}
}
package cn.itcast.classes;
/*
 *  Person类,当作方法的参数
 *  Person类型写在方法的参数列表中
 */
public class TestArguments {

	public static void main(String[] args) {
		//调用方法operatorPerson,传递Person类型对象
		Person p = new Person();
		operatorPerson(p);
		//传递有名对象p
		operatorPerson(new Person());
		//传递匿名对象，只能用一次,个人感觉，匿名对象可以理解为：定义了一个你不知道名字的类x，x用完就释放了
		//但是在operatorPerson内存调用过这个名字x，x.eat() x.run()
	}
	/*
	 *  方法operatorPerson,参数类型是Person类型
	 *  调用方法operatorPerson,必须传递Person类型的对象
	 */
	public static void operatorPerson(Person p){
		//可以使用引用类型p调用Person类的功能
		p.eat();
		p.run();
	}

}
output:
			张三  在吃饭
			张三 在跑步
			张三  在吃饭
			张三 在跑步
```
**3.3	抽象类作为方法参数与返回值**
	抽象类作为方法参数
a是抽象类Animal cat是抽象类的子类
传参过程  方法：operatorAnimal(Animal a)，用到的是父类抽象类，传递进来的是子类cat c（只有传cat才能有编译，因为Animal抽象没有主体）
其实就是多态：Animal a = new cat:
**这种方法的优点：扩展性，既可以传cat又可以传dog**条件：这两个都是Animal的子类，具有共同抽象方法eat
operatorAnimal(c); 
也可以；operatorAnimal( new Dog()); 也可以调用Dog的eat方法
```java
package cn.itcast.abstractclass;
/*
 *  将抽象类类型,作为方法的参数进行传递
 */
public class TestArguments {
	public static void main(String[] args) {
		//调用operatorAnimal,传递子类对象
		Cat c = new Cat();
		operatorAnimal(c);
		
		operatorAnimal( new Dog());
	}
	/*
	 *  方法operatorAnimal,参数是一个抽象类
	 *  调用方法,传递Animal类型对象,Animal抽象类没有对象
	 *  只能传递Animal的子类的对象 (多态)
	 *  可以传递Animal的任意的子类对象
	 */
	public static void operatorAnimal(Animal a){
		//引用变量a,调用方法eat
		a.eat();
	}
}
```
**	抽象类作为方法返回值

```java
public class GetAnimal {
	/*
	 * 定义方法,方法的返回值是Animal类型
	 * 抽象类,抽象类没有对象的,因此在方法的return后,返回Animal 的子类的对象
	 */
	public Animal getAnimal(int i){ //要分清类的定义和方法的定义，方法的定义必须有返回值
		if(i==0)
			
			return new Cat();
		
		return new Dog();
	}
}
public class TestReturn {
	public static void main(String[] args) {
		//调用GetAnimal类的方法,getAnimal,返回值是Animal
		GetAnimal g = new GetAnimal();
		
		Animal a= g.getAnimal(9);//方法的返回了Animal类型,return new Cat() 
		//getAnimal 返回的是Animal类，因此不能用cat类作为接收
		//Cat a = g.getAnimal(9);  这是错误的
		//问题：Animal不是抽象类吗，它还能eat？
		//答案是肯定的，因为返回的类型虽然是Animal，但是return的是他的子类Dog或者Cat
		//因此：这种方法可以理解为：Animal a= g.getAnimal(9);  等价于   Animal a = new（Cat）/new（Dog）
		//好处在于：可以通过给getAnimal设定参数，选择返回的类型是Cat或者Dog，比之前死的返回这一种要好用
		a.eat();
	}
}

```


## Python抽象类和抽象方法的定义

```python
from abc import   ABCmeta, abstractmethod

class People(metaclass=ABCmeta): #创建抽象类
    def __init__(self):
        pass

    @abstractmethod   #定义抽象方法
    def introduce_yourself(self):
        pass

    @abstractmethod
    def say(self):
        print("I'm like you！！")


class Student(People):
    def __init__(self, name, age, sex):
        self.name = name
        self.age = age
        self.sex = sex

    def say(self): 
        print("hello world")

    def introduce_yourself(self):
        print("Name is {}, age is {}, sex is {} ".format(self.name, self.age, self.sex))
        
	stu = Student("nlj", 30, "female")
	stu.say()       
```

**3.4	接口作为方法参数与返回值**
	接口作为方法参数
接口作为方法参数的情况是很常见的，经常会碰到。当遇到方法参数为接口类型时，那么该方法要传入一个接口实现类对象。如下代码演示。

```java
//接口
interface Smoke{
	public abstract void smoking();
}
//接口定义抽象方法:    interface Smoke{
//创建子类实现接口Smoke:    class Student implements Smoke{
//创建测试类:public class Test {
//在测试类中创建方法调用子类实现接口 public static void method(Smoke sm){//接口作为参数
//实际是通过sm变量调用smoking方法，这时实际调用的是Student对象中的smoking方法
//问题：如果有两个子类实现了Smoke接口的抽象方法，那么效果如何呢？
//答案：取决于你的Test中定义的子类属于哪一个实现类 STudent ？或者Student1
//Smoke s = new Student();//调用method方法，method(s);
class Student implements Smoke{
	@Override
	public void smoking() {
		System.out.println("别抽烟");
	}
}
class Student1 implements Smoke{
	@Override
	public void smoking() {
		System.out.println("老子就抽烟");
	}
}
//测试类
public class Test {
	public static void main(String[] args) {
		//通过多态的方式，创建一个Smoke类型的变量，而这个对象实际是Student
		Smoke s = new Student();
		//调用method方法
		method(s); //等价于 method(new Student())  利用匿名对象
		Smoke a = new Student1(); 
		method(a);
	}
	
	//定义一个方法method，用来接收一个Smoke类型对象，在方法中调用Smoke对象的show方法
	public static void method(Smoke sm){//接口作为参数
		//通过sm变量调用smoking方法，这时实际调用的是Student对象中的smoking方法
		sm.smoking();
	}
}
输出:
	别抽烟
	老子就抽烟
```


**3.4	接口作为方法参数与返回值**
	接口作为方法返回值
接口作为方法返回值的情况，在后面的学习中会碰到。当遇到方法返回值是接口类型时，那么该方法需要返回一个接口实现类对象。如下代码演示。
1、定义接口：interface Smoke｛
2、定义接口的实现类：class Student implements Smoke{
3、测试类中创建method方法  返回值类型是Smoke
```java
`Smoke s = method();` 
```
	//调用method方法的返回值类型是Smoke，接收方式是左边的方法
4、调用Smoke的方法s.smoking()
问题：怎么确定返回的Smoke接口中的重写方法对应的是Student?
答案：因为在Method中写出了调用的接口的实现类是Student，所以能够对应找到s.smoking()
如果你选择返回的是Student1的重写方法smoking，则需要修改method中的

```java
Smoke sm = new Student();→Smoke sm = new Student1();
```

```java
//接口
interface Smoke{
	public abstract void smoking();
}
class Student implements Smoke{
	@Override
	public void smoking() {
		System.out.println("课下吸口烟，赛过活神仙");
	}
}
class Student1 implements Smoke{
	@Override
	public void smoking() {
		System.out.println("哈哈哈哈哈哈哈");
	}
}
//测试类
public class Test {
	public static void main(String[] args) {
		//调用method方法，获取返回的会吸烟的对象
		Smoke s = method();
		//通过s变量调用smoking方法,这时实际调用的是Student对象中的smoking方法
		s.smoking();
	}
	
	//定义一个方法method，用来获取一个具备吸烟功能的对象，并在方法中完成吸烟者的创建
	public static Smoke method(){
		Smoke sm = new Student();
		return sm;
	}
}
```
下面的代码解析：
1、定义一个抽象类Animal，抽象类的抽象方法是eat() 一个返回值类型是Animal类静态方法getInstance()（实际上返回的是Cat子类，所以有eat的重写方法）
2、定义一个cat子类继承抽象类Animal
3、下面这个代码，表面上没有cat实际上实在执行cat类中的eat
4、import java.util.Calendar;导入日历类，这个类实际上是一个抽象类

```java
Animal a = Animal.getInstance();
		a.eat();
```

```java
package cn.itcast.demo03;

public abstract class Animal {
	public abstract void eat();
	
	/*
	 * 抽象类Animal,定义方法,返回值是Animal类型
	 * 抽象类没有对象,此方法方便调用,写为静态修饰
	 */
	public static Animal getInstance(){
		return new Cat();
	}
}


package cn.itcast.demo03;

public class Cat  extends Animal{
	public void eat(){
		System.out.println("猫吃鱼");
	}
}


package cn.itcast.demo03;

import java.util.Calendar;

public class Test {
	public static void main(String[] args) {
		//直接调用抽象类的静态方法getInstance获取抽象类的子类的对象
		//抽象类的静态方法,返回了自己的子类对象
		//对于调用者来讲: 不需要关注子类是谁
		Animal a = Animal.getInstance();
		a.eat();
		
		//日历类
		Calendar c = Calendar.getInstance();
		System.out.println(c);
	}
}
```
![星级酒店案例](https://img-blog.csdnimg.cn/20201201215827600.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQwMzEwNzEw,size_16,color_FFFFFF,t_70)
代码：day14 cn.itcast.hotel

接口是用来定义特殊特征（方法）的
继承抽象类 是用来重写共有特征（方法）的
厨师、服务员、经理的共有的 定义在abstract中
经理有一个private double money; 其他的都是public

## 第1章 Java的API及Object类

		在以前的学习过程中，我们都在学习对象基本特征、对象的使用以及对象的关系。接下来我们开始使用对象做事情，那么在使用对象做事情之前，我们要学习一些API中提供的常用对象。首先在学习API中的Object类之前,先来学习如何使用API。

**1.1	Java 的API**
Java 的API（API: Application(应用) Programming(程序) Interface(接口)）
Java API就是JDK中提供给我们使用的类，这些类将底层的代码实现封装了起来，我们不需要关心这些类是如何实现的，只需要学习这些类如何使用即可。
在JDK安装目录下有个src.zip文件，这个文件解压缩后里面的内容是所有Java类的源文件。可以在其中查看相对应的类的源码。
我们在每次查看类中的方法时，都打开源代码进行查看，这种方式过于麻烦。其实，我们可以通过查帮助文档的方式，来了解Java提供的API如何使用。如下图操作：查找Object类
**1.2	Object类概述**：
Object类是Java语言中的**根类**，即**所有类的父类**。
它中描述的所有方法子类都可以使用。所有类在创建对象的时候，最终找的父类就是Object。
在Object类众多方法中，我们先学习equals方法与toString方法，其他方法后面课程中会陆续学到。

**1.3	equals方法** ：作比较
equals方法，用于比较两个对象是否相同，它其实就是使用两个对象的内存地址在比较。Object类中的equals方法内部使用的就是==比较运算符。
在开发中要比较两个对象是否相同，经常会根据对象中的属性值进行比较，也就是在开发经常需要子类重写equals方法根据对象的属性值进行比较。如下代码演示：

```java
package cn.itcast.demo01;

public class Person extends Object{
	private String name;
	private int age;
	
	public Person(){}
	
	public Person(String name, int age) {
		this.name = name;
		this.age = age;
	}
	/*
	 * 重写父类的方法toString()
	 * 没有必要让调用者看到内存地址
	 * 要求: 方法中,返回类中所有成员变量的值
	 */
	public String toString(){
		return name + age;
	}
	
	
	/*
	 * 将父类的equals方法写过来,重写父类的方法
	 * 但是,不改变父类方法的源代码, 方法equals 比较两个对象的内存地址
	 * 
	 * 两个对象,比较地址,没有意义
	 * 比较两个对象的成员变量,age
	 * 两个对象变量age相同,返回true,不同返回false
	 * 
	 * 重写父类的equals,自己定义自己对象的比较方式
	 */
	public boolean equals(Object obj){
		if( this == obj){
			return true;
		}
		
		//对参数obj,非null判断
		if( obj == null){
			return false;
		}
		
		if( obj instanceof Person){
			// 参数obj接受到是Person对象,才能转型
			// 对obj参数进行类型的向下转型,obj转成Person类型
			Person p = (Person)obj;
			return this.age ==  p.age;
		}
		return false;
	}
	
	
	
	public String getName() {
		return name;
	}
	public void setName(String name) {
		this.name = name;
	}
	public int getAge() {
		return age;
	}
	public void setAge(int age) {
		this.age = age;
	}
	
	 
}

package cn.itcast.demo01;

public class TestEquals {
	public static void main(String[] args) {
		//Person类继承Object类,继承下来了父类的方法equals
		Person p1 = new Person("李四",20);
		Person p2 = new Person("张三",20);
		
      
		//Person对象p1,调用父类的方法equals,进行对象的比较
		boolean b = p1.equals(p1);
		System.out.println(b);
		boolean bo  = p1.equals(p2);
		System.out.println(bo);
	}
}
public boolean equals(Object obj){
		if( this == obj){
			return true;
		}    
//这个equals中this = obj比较的是地址，因为只要new一次输出的地址都不同，所以没什么意义，
//因此，作者重写了equals，比较年龄，名称不比较。代码如上所示
equals Peson类继承，重写了equals,源代码和父类的源代码是一样的
equals中的this指向的是调用者 p1  p1.equals(obj)  this == obj:比较的是内存地址，内存地址不同所以返回false
注意：因为equals的输入参数是object类的对象，所以需要进行一个向下转型，即，强转为Person类，这样才能进行age的比较
if( obj instanceof Person){这句话的意思是，首先obj是Person类，才能进行强转换
这句话是判断这个对象是不是Person类
```
**1.4	toString方法**

toString方法返回该对象的字符串表示，其实该字符串内容就是**对象的类型+@+内存地址值。**
由于toString方法返回的结果是内存地址，而在开发中，经常需要按照对象的属性得到相应的字符串表现形式，因此也需要重写它。
需要重写，不重写

> 这里是引用

返回的是内存地址
```java
class Person extends Object{
	int age ;
	//根据Person类的属性重写toString方法
	public String toString() {
		return "Person [age=" + age + "]";
	}
}

## 第2章 String类


## String方法调用集锦重要，看懂以下代码就可以

​```java
package cn.itcast.demo02;
/*
 *  String类的查找功能
 */
public class StringDemo4 {
	public static void main(String[] args) {
		function_9();
	}
	/*
	 *  boolean equals(Object obj)
	 *  方法传递字符串,判断字符串中的字符是否完全相同,如果完全相同返回true
	 *  
	 *  boolean equalsIgnoreCase(String s)
	 *  传递字符串,判断字符串中的字符是否相同,忽略大小写
	 */
	public static void function_9(){
		String str1 = "Abc";
		String str2 = "abc";
		//分别调用equals和equalsIgnoreCase
		boolean b1 = str1.equals(str2);
		boolean b2 = str1.equalsIgnoreCase(str2);
		System.out.println(b1);
		System.out.println(b2);
	}
	
	/*
	 * char[] toCharArray() 将字符串转成字符数组
	 * 功能和构造方法相反
	 */
	public static void function_8(){
		String str = "itcast";
		//调用String类的方法toCharArray()
		char[] ch = str.toCharArray();
		for(int i = 0 ; i < ch.length ; i++){
			System.out.println(ch[i]);
		}
输出：
i
t
c
a
s
t
	}
	
	/*
	 *  byte[] getBytes() 将字符串转成字节数组
	 *  此功能和String构造方法相反
	 *  byte数组相关的功能,查询编码表
	 */
	public static void function_7(){
		String str = "abc";
		//调用String类方法getBytes字符串转成字节数组
		byte[] bytes = str.getBytes();
		for(int i = 0 ; i < bytes.length ; i++){
			System.out.println(bytes[i]);//输出：97 98 99
		}
	}
	
	/*
	 *  int indexOf(char ch)
	 *  查找一个字符,在字符串中第一次出现的索引
	 *  被查找的字符不存在,返回-1
	 */
	public static void function_6(){
		String str = "itcast.cn";
		//调用String类的方法indexOf
		int index = str.indexOf('x');
		System.out.println(index);
	}
	
	/*
	 *  boolean contains (String s)
	 *  判断一个字符串中,是否包含另一个字符串
	 */
	public static void function_5(){
		String str = "itcast.cn";
		//调用String类的方法contains
		boolean b =str.contains("ac");
		System.out.println(b);
	}
	
	/*
	 * boolean endsWith(String prefix)
	 * 判断一个字符串是不是另一个字符串的后缀,结尾
	 * Demo.java
	 *     .java
	 */
	public static void function_4(){
		String str = "Demo.java";
		//调用String类方法endsWith
		boolean b = str.endsWith(".java");
		System.out.println(b);//输出：true
	}
	
	/*
	 * boolean startsWith(String prefix)  
	 * 判断一个字符串是不是另一个字符串的前缀,开头
	 * howareyou
	 * hOw
	 */
	  public static void function_3(){
		  String str = "howareyou";
		  //调用String类的方法startsWith
		  boolean b = str.startsWith("hOw");
		  System.out.println(b);//测试是否以hOw开始，输出false
	  }
	
	/*
	 *  String substring(int beginIndex,int endIndex) 获取字符串的一部分
	 *  返回新的字符串
	 *  包含头,不包含尾巴
	 *  
	 *  String substring(int beginIndex)获取字符串的一部分
	 *  包含头,后面的字符全要
	 */
	public static void function_2(){
		String str = "howareyou";
		//调用String类方法substring获取字符串一部分
		str= str.substring(1, 5);//字符串定义之后不能修改，所以打印结果还是str
		System.out.println(str);//输出：owar
		
		String str2 = "HelloWorld";
		str2 = str2.substring(1);//截取操作需要另一个字符串去接收，只有一个int输入：包含头，后面的字符全都要
		System.out.println(str2);//输出；elloWorld
	}
	
	/*
	 *  int length() 返回字符串的长度
	 *  包含多少个字符
	 */
	public static void function(){
		String str = "cfxdf#$REFewfrt54GT";
		//调用String类方法length,获取字符串长度
		int length = str.length();
		System.out.println(length);//输出19

	}
}
```


**2.1	String类的概述**
查阅API中的String类的描述，发现String 类代表字符串。Java 程序中的所有字符串字面值（如 "abc" ）都作为此类的实例实现。

```java
 *   String类特点:
 *     一切都是对象,字符串事物 "" 也是对象
 *     类是描述事物,String类,描述字符串对象的类
 *     所有的 "" 都是String类的对象
 *     
 *     字符串是一个常量,一旦创建,不能改变
//演示字符串
String str  = "itcast";  //"itcast"是对象  str是引用类型的变量  不需要写new
//String也是object子类
str = "传智播客";
```
继续查API发现，字符串有大量的重载的构造方法。通过String类的构造方法可以完成字符串对象的创建，那么，通过使用双引号的方式创建对象与new的方式创建对象，有什么不同呢？看如下程序与图解：
```java
String s3 = "abc";
String s4 = new String("abc");
System.out.println(s3==s4);//false
System.out.println(s3.equals(s4));//true,
//因为String重写了equals方法，建立了字符串自己的判断相同的依据（通过字符串对象中的字符来判断）
```

```java
public class StringDemo2 {
	public static void main(String[] args) {
		//字符串定义方式2个, 直接=  使用String类的构造方法
		String str1 = new String("abc");
		String str2 = "abc";
		System.out.println(str1);
		System.out.println(str2);
		
		System.out.println(str1==str2);//引用数据类型,比较对象的地址 false
		System.out.println(str1.equals(str2));//true
	}
}
// 输出：  字符串.equals比较的是对象每隔字符是否相同（String是Object的子类，因此String重写了equals）

abc
abc
false
true

'''
s3和s4的创建方式有什么不同呢？
	s3创建，在内存中只有一个对象。这个对象在字符串常量池中
	s4创建，在内存中有两个对象。一个new的对象在堆中，一个字符串本身对象，在字符串常量池中

​```java
package cn.itcast.demo02;
/*
 *   String类特点:
 *     一切都是对象,字符串事物 "" 也是对象
 *     类是描述事物,String类,描述字符串对象的类
 *     所有的 "" 都是String类的对象
 *     
 *     字符串是一个常量,一旦创建,不能改变
 */
public class StringDemo {
	public static void main(String[] args) {
		//引用变量str执行内存变化
		//定义好的字符串对象,不变
		String str = "itcast";
		System.out.println(str);
		str = "itheima";
		System.out.println(str);
		
		
	}
}
```
**2.2	String类构造方法**   有很多构造方法
构造方法是用来完成String对象的创建，下图中给出了一部分构造方法需要在API中找到，并能够使用下列构造方法创建对象。
day15 StringDemo3
```java
package cn.itcast.demo02;
 /*
  *  String类构造方法
  *  String类的构造方法,重载形式
  * 
  */
public class StringDemo3 {
	public static void main(String[] args) {
		function_1();
	}
	/*
	 * String(char[] value) 传递字符数组
	 * 将字符数组,转成字符串, 字符数组的参数,不查询编码表
	 * 
	 * String(char[] value, int offset, int count) 传递字符数组
	 * 将字符数组的一部分转成字符串
	 * offset  数组开始索引
	 * count   个数
	 */
	public static void function_1(){
		char[] ch = {'a','b','c','d','e','f'};
		//调用String构造方法,传递字符数组
		String s = new String(ch);
		System.out.println(s);
		
		String s1 = new String(ch,1,4);
		System.out.println(s1);
	}
	
	
	/*
	 *  定义方法,String类的构造方法
	 *  String(byte[] bytes)  传递字节数组
	 *  字节数组转成字符串
	 *  通过使用平台的默认字符集解码指定的 byte 数组，构造一个新的 String。
	 *  平台 : 机器操作系统
	 *  默认字符集: 操作系统中的默认编码表, 默认编码表GBK
	 *  将字节数组中的每个字节,查询了编码表,得到的结果
	 *  字节是负数,汉字的字节编码就是负数, 默认编码表 ,一个汉字采用2个字节表示
	 * 即：｛-97，-98｝这表示一个汉字
	 *  
	 *  String(byte[] bytes, int offset, int length) 传递字节数组
	 *  字节数组的一部分转成字符串
	 *  offset 数组的起始的索引
	 *  length 个数,转几个   , 不是结束的索引
	 */
	public static void function(){
		byte[] bytes = {97,98,99,100};
		//调用String类的构造方法,传递字节数组
		String s = new String(bytes);
		System.out.println(s);
		
		byte[] bytes1 ={65,66,67,68,69};//对应ABCDE
		//调用String构造方法,传递数组,传递2个int值
		String s1 = new String(bytes1,0,2);//0是起始位置，2是几个
		System.out.println(s1);
	}
}
```



```java
String s1 = new String(); //创建String对象，字符串中没有内容
	
	byte[] bys = new byte[]{97,98,99,100};
	String s2 = new String(bys); // 创建String对象，把数组元素作为字符串的内容
	String s3 = new String(bys, 1, 3); //创建String对象，把一部分数组元素作为字符串的内容，参数offset为数组元素的起始索引位置，参数length为要几个元素
	
	char[] chs = new char[]{’a’,’b’,’c’,’d’,’e’};
	String s4 = new String(chs); //创建String对象，把数组元素作为字符串的内容
	String s5 = new String(chs, 0, 3);//创建String对象，把一部分数组元素作为字符串的内容，参数offset为数组元素的起始索引位置，参数count为要几个元素
	String s6 = new String(“abc”); //创建String对象，字符串内容为abc
```

```

**2.3	String类的方法查找**
String类中有很多的常用的方法，我们在学习一个类的时候，不要盲目的把所有的方法尝试去使用一遍，这时我们应该根据这个对象的特点分析这个对象应该具备那些功能，这样大家应用起来更方便。
字符串是一个对象，那么它的方法必然是围绕操作这个对象的数据而定义的。我们想想字符串中有哪些功能呢？
	字符串中有多少个字符?

​```java
String str = "abcde";
int len = str.length();
System.out.println("len="+len);
```
	获取部分字符串。

```java
String str = "abcde";
String s1 = str.substring(1); //返回一个新字符串，内容为指定位置开始到字符串末尾的所有字符
String s2 = str.substring(2, 4);//返回一个新字符串，内容为指定位置开始到指定位置结束所有字符
System.out.println("str="+str);
System.out.println("s1="+s1);
System.out.println("s2="+s2);
```

## 3.1 StringBuffer类

在学习String类时，API中说字符串缓冲区支持可变的字符串，什么是字符串缓冲区呢？接下来我们来研究下字符串缓冲区。
查阅StringBuffer的API，**StringBuffer又称为可变字符序列**，它是一个类似于 String 的字符串缓冲区，通过某些方法调用可以**改变该序列的长度和内容**。
原来**StringBuffer是个字符串的缓冲区**，即就是它是一个容器，容器中可以装很多字符串。并且能够对其中的字符串进行各种操作。

StringBuffer一系列基本操作：
看懂基础代码及调用格式即可

.append：将指定的字符串追加到此字符序列
.delete：移除次序列的子字符串的字符
.insert：将字符串插入此字符序列
.replace：使用给定String中的字符替换此序列的子字符串中的字符
.reverse：将此字符串序列用其反转形式取代
.toString：返回此序列中数据的字符串表示形式
```java
package cn.itcast.demo03;

public class StringBufferDemo {
	public static void main(String[] args) {
		function_5();
	}
	/*
	 *  StringBuffer类的方法
	 *   String toString() 继承Object,重写toString()
	 *   将缓冲区中的所有字符,变成字符串
	 */
	public static void function_5(){
		StringBuffer buffer = new StringBuffer();
		buffer.append("abcdef");
		buffer.append(12345);
		
		//将可变的字符串缓冲区对象,变成了不可变String对象
		String s = buffer.toString();
		System.out.println(s);
	}
	//输出：abcdef12345
	/*
	 *  StringBuffer类的方法
	 *    reverse() 将缓冲区中的字符反转
	 */
	public static void function_4(){
		StringBuffer buffer = new StringBuffer();
		buffer.append("abcdef");
		
		buffer.reverse();
		
		System.out.println(buffer);
	}
	//输出：fedcba
	/*
	 *  StringBuffer类方法
	 *    replace(int start,int end, String str)
	 *    将指定的索引范围内的所有字符,替换成新的字符串
	 */
	public static void function_3(){
		StringBuffer buffer = new StringBuffer();
		buffer.append("abcdef");
		
		buffer.replace(1, 4, "Q");
	// 将1到4索引（四个字符）替换成一个Q
		System.out.println(buffer);
	}
	//输出：aQef
	/*
	 *  StringBuffer类方法 insert
	 *    insert(int index, 任意类型)
	 *  将任意类型数据,插入到缓冲区的指定索引上
	 */
	 public static void function_2(){
		 StringBuffer buffer = new StringBuffer();
		 buffer.append("abcdef");	 
		 
		 buffer.insert(3, 9.5);//在3索引处增加一个9.5
		 System.out.println(buffer);
	 }
	//abc9.5def
	/*
	 * StringBuffer类方法
	 *   delete(int start,int end) 删除缓冲区中字符
	 *   开始索引包含,结尾索引不包含
	 */
	public static void function_1(){
		StringBuffer buffer = new StringBuffer();
		buffer.append("abcdef");
		
		buffer.delete(1,5);//从1索引开始删除5个
		System.out.println(buffer);
	}
	//输出：af
	
	/*
	 *  StringBuffer类方法
	 *   StringBuffer append, 将任意类型的数据,添加缓冲区
	 *   append 返回值,写return this
	 *   调用者是谁,返回值就是谁
	 */
	public static void function(){
		StringBuffer buffer = new StringBuffer();
		//调用StringBuffer方法append向缓冲区追加内容
		buffer.append(6).append(false).append('a').append(1.5);
		System.out.println(buffer);
	}
	//输出；6falsea1.5
}

```
**代码任务描述：**
任务：将一个int[]中元素转成字符串 
输入：int[] arr = {34,12,89,68};
输出：格式 [34,12,89,68]

```java
package cn.itcast.demo03;

public class StringBufferTest {
	public static void main(String[] args) {
		int[] arr = {4,1,4,56,7,8,76};
		System.out.println(arr);
		//为什么这样打印的是：[I@1d1e730
		System.out.println(toString(arr));
	}
   /*
    * 任务：将一个int[]中元素转成字符串 
    * 输入：int[] arr = {34,12,89,68};
    * 输出：格式 [34,12,89,68]
    * String s = "["
    * 数组遍历
    *   s+= arr[i];
    *  s+"]"
    *  StringBuffer实现,节约内存空间, String + 在缓冲区中,append方法
    */
	public static String toString(int[] arr){
		//创建字符串缓冲区
		StringBuffer buffer = new StringBuffer();
		buffer.append("[");
		//数组遍历
		for(int i = 0 ; i < arr.length;i++){
			//判断是不是数组的最后一个元素
			if(i == arr.length-1){
				buffer.append(arr[i]).append("]");
			}else{
				buffer.append(arr[i]).append(",");
			}
		}
		return buffer.toString();
	}
}

```

## 1.3 正则表达式规则匹配练习
syso   alt+/
**1.1	正则表达式的概念**
	正则表达式（英语：Regular Expression，在代码中常简写为regex）。
	正则表达式是一个字符串，使用单个字符串来描述、用来定义匹配规则，匹配一系列符合某个句法规则的字符串。在开发中，正则表达式通常被用来检索、替换那些符合某个规则的文本。
	
**1.2	正则表达式的匹配规则**
	参照帮助文档，在Pattern类中有正则表达式的的规则定义，正则表达式中明确区分大小写字母。我们来学习语法规则。
	正则表达式的语法规则：
	字符：x
	含义：代表的是字符x
	例如：匹配规则为 "a"，那么需要匹配的字符串内容就是 ”a”
	
	字符：\\    翻译：将转义字符转义为普通斜线，失去了转义字符的意思
	含义：代表的是反斜线字符'\'
	例如：匹配规则为"\\" ，那么需要匹配的字符串内容就是 ”\”
	
	字符：\t
	含义：制表符
	例如：匹配规则为"\t" ，那么对应的效果就是产生一个制表符的空间
	
	字符：\n
	含义：换行符
	例如：匹配规则为"\n"，那么对应的效果就是换行,光标在原有位置的下一行
	
	字符：\r
	含义：回车符
	例如：匹配规则为"\r" ，那么对应的效果就是回车后的效果,光标来到下一行行首
	
	字符类：[abc]
	含义：代表的是字符a、b 或 c
	例如：匹配规则为"[abc]" ，那么需要匹配的内容就是字符a，或者字符b，或字符c的一个
	
	字符类：[^abc]
	含义：代表的是除了 a、b 或 c以外的任何字符
	例如：匹配规则为"[^abc]"，那么需要匹配的内容就是不是字符a，或者不是字符b，或不是字符c的任意一个字符
	
	字符类：[a-zA-Z]
	含义：代表的是a 到 z 或 A 到 Z，两头的字母包括在内
	例如：匹配规则为"[a-zA-Z]"，那么需要匹配的是一个大写或者小写字母
	
	字符类：[0-9]
	含义：代表的是 0到9数字，两头的数字包括在内
	例如：匹配规则为"[0-9]"，那么需要匹配的是一个数字
	
	字符类：[a-zA-Z_0-9]
	含义：代表的字母或者数字或者下划线(即单词字符)
	例如：匹配规则为" [a-zA-Z_0-9] "，那么需要匹配的是一个字母或者是一个数字或一个下滑线
	
	预定义字符类：.
	含义：代表的是任何字符
	例如：匹配规则为" . "，那么需要匹配的是一个任意字符。如果，就想使用 . 的话，使用匹配规则"\\."来实现
	
	预定义字符类：\d
	含义：代表的是 0到9数字，两头的数字包括在内，相当于[0-9]
	例如：匹配规则为"\d "，那么需要匹配的是一个数字
	
	预定义字符类：\w
	含义：代表的字母或者数字或者下划线(即单词字符)，相当于[a-zA-Z_0-9]
	例如：匹配规则为"\w "，，那么需要匹配的是一个字母或者是一个数字或一个下滑线
	
	边界匹配器：^
	含义：代表的是行的开头
	例如：匹配规则为^[abc][0-9]$ ，那么需要匹配的内容从[abc]这个位置开始, 相当于左双引号
	
	边界匹配器：$
	含义：代表的是行的结尾
	例如：匹配规则为^[abc][0-9]$ ，那么需要匹配的内容以[0-9]这个结束, 相当于右双引号
	
	边界匹配器：\b
	含义：代表的是单词边界
	例如：匹配规则为"\b[abc]\b" ，那么代表的是字母a或b或c的左右两边需要的是非单词字符([a-zA-Z_0-9])
	
	数量词：X?
	含义：代表的是X出现一次或一次也没有
	例如：匹配规则为"a?"，那么需要匹配的内容是一个字符a，或者一个a都没有
	
	数量词：X*
	含义：代表的是X出现零次或多次
	例如：匹配规则为"a*" ，那么需要匹配的内容是多个字符a，或者一个a都没有
	
	数量词：X+
	含义：代表的是X出现一次或多次
	例如：匹配规则为"a+"，那么需要匹配的内容是多个字符a，或者一个a
	
	数量词：X{n}
	含义：代表的是X出现恰好 n 次
	例如：匹配规则为"a{5}"，那么需要匹配的内容是5个字符a
	
	数量词：X{n,}
	含义：代表的是X出现至少 n 次
	例如：匹配规则为"a{5, }"，那么需要匹配的内容是最少有5个字符a
	
	数量词：X{n,m}
	含义：代表的是X出现至少 n 次，但是不超过 m 次
	例如：匹配规则为"a{5,8}"，那么需要匹配的内容是有5个字符a 到 8个字符a之间
**1.3	正则表达式规则匹配练习**
请写出满足如下匹配规则的字符串:
**规则："[0-9]{6,12}"
该规则需要匹配的内容是：长度为6位到12位的数字。**
如：使用数据"123456789"进行匹配结果为true；
使用数据"12345"进行匹配结果为false。

**规则："1[34578][0-9]{9}"
该规则需要匹配的内容是：11位的手机号码，第1位为1，第2位为3、4、5、7、8中的一个，后面9位为0到9之间的任意数**字。
如：使用数据"12345678901"进行匹配结果为false；
使用数据"13312345678"进行匹配结果为true。

规则："a*b"
该规则需要匹配的内容是：在多个a或零个a后面有个b；b必须为最后一个字符。
如：使用数据"aaaaab"进行匹配结果为true；
使用数据"abc"进行匹配结果为false。

```java
package cn.itcast.demo01;
/*
 *  实现正则规则和字符串进行匹配,使用到字符串类的方法
 *  String类三个和正则表达式相关的方法
 *    boolean matches(String 正则的规则)
 *    "abc".matches("[a]")  匹配成功返回true
 *    
 *    String[] split(String 正则的规则)
 *    "abc".split("a") 使用规则将字符串进行切割
 *     
 *    String replaceAll( String 正则规则,String 字符串)
 *    "abc0123".repalceAll("[\\d]","#")
 *    按照正则的规则,替换字符串
 */ 
public class RegexDemo {
	public static void main(String[] args) {
		checkTel();
	}
	
	
	/*
	 *  检查手机号码是否合法
	 *  1开头 可以是34578  0-9 位数固定11位
	 */
	public static void checkTel(){
		String telNumber = "1335128005";
		//String类的方法matches
		boolean b = telNumber.matches("1[34857][\\d]{9}");
		//为什么第二位不需要空格或者逗号呢，记住吧
		System.out.println(b);
	}
	
	/*
	 *  检查QQ号码是否合法
	 *  0不能开头,全数字, 位数5,10位
	 *  123456 
	 *  \\d  \\D匹配不是数字
	 */
	public static void checkQQ(){
		String QQ = "123456";
		//检查QQ号码和规则是否匹配,String类的方法matches
		boolean b = QQ.matches("[1-9][\\d]{4,9}");   // \d  表示0到9  那为什么这里用\\d呢
		//原因：\\d，第一个\转义第二个\,这样才表示\d
		System.out.println(b);
	}
}


```


```java
package cn.itcast.demo01;

public class RegexDemo1 {
	public static void main(String[] args) {
		replaceAll_1();
	}
	
	/*
	 * "Hello12345World6789012"将所有数字替换掉
	 * String类方法replaceAll(正则规则,替换后的新字符串)
	 */
	public static void replaceAll_1(){
		String str = "Hello12345World6789012";
		str = str.replaceAll("[\\d]+", "#");//将每个数字串改成#
		//str = str.replaceAll("[\\d]", "#");将每个数字改成#
		System.out.println(str);
	}
输出：Hello#World#

	/*
	 * String类方法split对字符串进行切割
	 * 192.168.105.27 按照 点切割字符串
	 */
	public static void split_3(){
		String ip = "192.168.105.27";
		String[] strArr = ip.split("\\.");//转移成普通的.，否则输出的是空的数组
		System.out.println("数组的长度"+strArr.length);
		for(int i = 0 ; i < strArr.length ; i++){
			System.out.println(strArr[i]);
		}
	}
输出：数组的长度4
192
168
105
27

	/*
	 * String类方法split对字符串进行切割
	 * 18 22 40 65 按照空格切割字符串
	 */
	public static void split_2(){
		String str = "18    22     40          65";
		String[] strArr = str.split(" +");  //+代表的是空格出现一次或多次
		System.out.println("数组的长度"+strArr.length);
		for(int i = 0 ; i < strArr.length ; i++){
			System.out.println(strArr[i]);
		}
	}
输出：数组的长度4
18
22
40
65

	/*
	 *  String类方法split对字符串进行切割
	 *  12-25-36-98  按照-对字符串进行切割
	 */
	public static void split_1(){
		String str = "12-25-36-98";
		//按照-对字符串进行切割,String类方法split
		String[] strArr = str.split("-");
		System.out.println("数组的长度"+strArr.length);
		for(int i = 0 ; i < strArr.length ; i++){
			System.out.println(strArr[i]);
		}
   输出：数组的长度4
		12
		25
		36
		98
	}
}

```

1.5	正则表达式练习
	匹配正确的数字
匹配规则：
	匹配正整数：”\\d+”
	匹配正小数：”\\d+\\.\\d+”  
	匹配负整数：”-\\d+”
	匹配负小数：”-\\d+\\.\\d+”
	匹配保留两位小数的正数：”\\d+\\.\\d{2}”
	匹配保留1-3位小数的正数：”\\d+\\.\\d{1,3}”

	匹配合法的邮箱
匹配规则：
	”[a-zA-Z_0-9]+@[a-zA-Z_0-9]+(\\.[a-zA-Z_0-9]+)+”
	”\\w+@\\w+(\\.\\w+)+”

	获取IP地址(192.168.1.100)中的每段数字
匹配规则：
	”\\.”


```java
package cn.itcast.demo01;

public class RegexDemo2 {
	public static void main(String[] args) {
		checkMail();
	}
	/*
	 *  检查邮件地址是否合法
	 *  规则:
	 *   1234567@qq.com
	 *   mym_ail@sina.com
	 *   nimail@163.com
	 *   wodemail@yahoo.com.cn    
	 *   
	 *   @: 前  数字字母_ 个数不能少于1个
	 *   @: 后  数字字母     个数不能少于1个
	 *   .: 后面 字母 
	 *     
	 */
	public static void checkMail(){
		String email ="abc123@sina.com";
		boolean b = email.matches("[a-zA-Z0-9_]+@[0-9a-z]+(\\.[a-z]+)+");
		 //含义：代表的字母或者数字或者下划线(即单词字符)，看不懂就去上面查字符类
		System.out.println(b);
	}
}
```

## 第2章 Date
**2.1	Date类概述**
类 Date 表示特定的瞬间，精确到毫秒。
继续查阅Date类的描述，发现Date拥有多个构造函数，只是部分已经过时，但是其中有未过时的构造函数可以把毫秒值转成日期对象。

//创建日期对象，把当前的毫秒值转成日期对象
Date date = new Date(1607616000000L);
System.out.println(date);
//打印结果：Fri Dec 11 00:00:00 CST 2020
可是将毫秒值转成日期后，输出的格式不利于我们阅读，继续查阅API，Date中有getYear、getMouth等方法，可以他们已经过时，继续往下查阅，看到了toString方法。

点开toString()方法查阅，原来上面打印的date对象就是默认调用了这个toString方法，并且在这个方法下面还有让我们参见toLocaleString方法，点进去，这个方法又过时了，从 JDK 1.1 开始，由 DateFormat.format(Date date) 取代。 
既然这个方法被DateFormat.format(Date date) 取代，那么就要去查阅DateFormat类。

```java
package cn.itcast.demo02;

import java.util.Date;

/*
 *  时间和日期类
 *    java.util.Date
 *    
 *  毫秒概念: 1000毫秒=1秒
 *  
 *  毫秒的0点: 
 *     System.currentTimeMillis() 返回值long类型参数
 *     获取当前日期的毫秒值   3742769374405
 *     时间原点; 公元1970年1月1日,午夜0:00:00 英国格林威治  毫秒值就是0
 *     时间2088年8月8日
 *  
 *  重要: 时间和日期的计算,必须依赖毫秒值
 *    XXX-XXX-XX = 毫秒
 *    
 * 		long time = System.currentTimeMillis();
		System.out.println(time);
 */
public class DateDemo {
	public static void main(String[] args) {
		function_3();
	}
	/*
	 * Date类方法 setTime(long )传递毫秒值
	 * 将日期对象,设置到指定毫秒值上
	 * 毫秒值转成日期对象
	 * 输出：Sun Dec 06 22:51:28 CST 2020
	   Thu Jan 01 08:00:00 CST 1970

	 * Date的构造方法
	 */
	public static void function_3(){
		Date date = new Date();
		System.out.println(date);
		
		date.setTime(0);
		System.out.println(date);
	}
	
	/*
	 *   Date类方法 getTime() 返回值long
	 *   返回的是毫秒值
	 *   将Date表示的日期,转成毫秒值
	 *   输出：1607266158056
	 *   日期和毫秒值转换
	 */
	public static void function_2(){
		Date date = new Date();
		long time = date.getTime();//将data表示的日期转换为毫秒值
		System.out.println(time);
	}
	
	/*
	 * Date类的long参数的构造方法
	 * Date(long ) 表示毫秒值
	 * 传递毫秒值,将毫秒值转成对应的日期对象
	 * 输出： Thu Jan 01 08:00:00 CST 1970   传递一个毫秒值，将毫秒值转换为对应的日期，Java从1970 00 00 00 开始（这个时区和格林威治时间有八个小时的时差）
	 */
	public static void function_1(){
		Date date = new Date(0);
		System.out.println(date);
	}
	
	/*
	 * Date类空参数构造方法
	 * 获取到的是,当前操作系统中的时间和日期
	 *输出： Sun Dec 06 22:47:39 CST 2020
	 */
	public static void function(){
		Date date = new Date();
		System.out.println(date);
	}
}

```

## 第3章 DateFormat

**3.1	DateFormat类概述**
DateFormat 是日期/时间格式化子类的抽象类，它以与语言无关的方式格式化并解析日期或时间。日期/时间格式化子类（如 SimpleDateFormat类）允许进行格式化（也就是日期 -> 文本）、解析（文本-> 日期）和标准化。
我们通过这个类可以帮我们完成日期和文本之间的转换。
继续阅读API，DateFormat 可帮助进行格式化并解析任何语言环境的日期。对于月、星期，甚至日历格式（阴历和阳历），**其代码可完全与语言环境的约定无关。
3.2	日期格式**
要格式化一个当前语言环境下的日期也就是日期 -> 文本），要通过下面的方法来完成。DateFormat是抽象类，我们需要使用其子类SimpleDateFormat来创建对象。
	构造方法

	DateFormat类方法

代码演示：
//创建日期格式化对象,在获取格式化对象时可以指定风格
DateFormat df= new SimpleDateFormat("yyyy-MM-dd");//对日期进行格式化
Date date = new Date(1607616000000L);
String str_time = df.format(date);
System.out.println(str_time);//2020年12月11日
	DateFormat类的作用：即可以将一个Date对象转换为一个符合指定格式的字符串，也可以将一个符合指定格式的字符串转为一个Date对象。
指定格式的具体规则我们可参照SimpleDateFormat类的说明，这里做简单介绍，规则是在一个字符串中，会将以下字母替换成对应时间组成部分，剩余内容原样输出：
	当出现y时，会将y替换成年
	当出现M时，会将M替换成月
	当出现d时，会将d替换成日
	当出现H时，会将H替换成时
	当出现m时，会将m替换成分
	当出现s时，会将s替换成秒
**3.3	DateFormat类常用方法**

	format方法，用来将Date对象转换成String
	parse方法，用来将String转换成Date（转换时，该String要符合指定格式，否则不能转换）。
代码演示：
练习一：把Date对象转换成String
     Date date = new Date(1607616000000L);//Fri Dec 11 00:00:00 CST 2020
	DateFormat df = new SimpleDateFormat(“yyyy年MM月dd日”);
	String str = df.format(date);
	//str中的内容为2020年12月11日

练习二：把String转换成Date对象
	String str = ”2020年12月11日”;
	DateFormat df = new SimpleDateFormat(“yyyy年MM月dd日”);
	Date date = df.parse( str );
	//Date对象中的内容为Fri Dec 11 00:00:00 CST 2020
**日期格式化**format
```java
package cn.itcast.demo02;

import java.text.SimpleDateFormat;
import java.util.Date;

/*
 *  对日期进行格式化 (自定义)
 *    对日期格式化的类 java.text.DateFormat 抽象类, 普通方法,也有抽象的方法
 *    实际使用是子类 java.text.SimpleDateFormat 可以使用父类普通方法,重写了抽象方法
 */
public class SimpleDateFormatDemo {
	public static void main(String[] args) {
		function();
	}
	/*
	 * 如何对日期格式化
	 *  步骤:
	 *    1. 创建SimpleDateFormat对象
	 *       在类构造方法中,写入字符串的日期格式 (自己定义)
	 *    2. SimpleDateFormat调用方法format对日期进行格式化
	 *         String format(Date date) 传递日期对象,返回字符串
	 *    日期模式:
	 *       yyyy    年份
	 *       MM      月份
	 *       dd      月中的天数
	 *       HH       0-23小时
	 *       mm      小时中的分钟
	 *       ss      秒
	 *       yyyy年MM月dd日 HH点mm分钟ss秒  汉字修改,: -  字母表示的每个字段不可以随便写
	 */
	public static void function(){
		SimpleDateFormat sdf = new SimpleDateFormat("yyyy年MM月dd日 HH点mm分钟ss秒");
		String date = sdf.format(new Date());
		System.out.println(date);
	}
}

```
**字符串转为日期对象**用parse
```java
package cn.itcast.demo02;

import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;

/*
 *   DateFormat类方法 parse
 *   将字符串解析为日期对象
 *   Date parse(String s) 字符串变成日期对象
 *   String => Date parse
 *   Date => String format
 *   
 */
public class SimpleDateFormatDemo1 {
	public static void main(String[] args) throws Exception{
		function();
	}
	/*
	 *  将字符串转成Date对象
	 *  DateFormat类方法 parse
	 *  步骤:
	 *    1. 创建SimpleDateFormat的对象
	 *       构造方法中,指定日期模式
	 *    2. 子类对象,调用方法 parse 传递String,返回Date
	 *    
	 *    注意: 时间和日期的模式yyyy-MM-dd, 必须和字符串中的时间日期匹配
	 *                     1995-5-6
	 *    
	 *    但是,日期是用户键盘输入, 日期根本不能输入
	 *    用户选择的形式
	 */
	public static void function() throws Exception{
		SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd");
		Date date = sdf.parse("1995-5-6");
		System.out.println(date);
	}
}

```

## 1、基本类型包装类
用户的输入数据都是字符串类型，然而程序开发过程中需要把字符串数据，根据需求转换成指定的基本数据类型，如年龄需要转换成int类型，考试成绩转为double类型
**基本类型包装类**实现的就是将字符串转换成指定的数据类型

8种基本类型对应的包装类
字节型：Byte             byte           
短整型：Short            short
整型：Integer             int
长整型：Long             long
字符型：Character     char
布尔型：Boolean        boolean
浮点型：Float             float
浮点型：Double         double

看懂下边的示例代码：
```java
package cn.itcast.demo1;
/*
 * Integer类,封装基本数据类型int,提高大量方法
 * 将字符串转成基本数据类型int
 * 
 * 
 * Integer i = Integer.valueOf("1");
 * i.intValue()
 */
public class IntegerDemo {
	public static void main(String[] args) {
		function_3();
	}
	/*
	 *  Integer类构造方法
	 *   Integer (String s)
	 *   将数字格式的字符串,传递到Integer类的构造方法中
	 *   创建Integer对象,包装的是一个字符串
	 *   将构造方法中的字符串,转成基本数据类型,调用方法,非静态的, intValue()
	 * 	 输出： 99  构造方法的格式  in.intValue()转为int类型
	 */
	public static void function_3(){
		Integer in = new Integer("100");
		int i = in.intValue();
		System.out.println(--i);
	}
	
	/*
	 *  如何将基本类型int,变成字符串
	 *  
	 *  int => String  任何类型+"" 变成String类型
	 *  Integer类中的静态方法toString()
	 *  
	 *  toString(int ,int 进制), 将int整数,转成指定进制数
	 *  0-9A-Z
	 */
	public static void function_2(){
		int i = 3;
		String s = i+"";
		System.out.println(s+1); //输出31这个31是字符串，不是数字32
		
		String s1 = Integer.toString(5,2); //将5转为二进制数 输出：101
		System.out.println(s1);
	}
	
	
	/*
	 * Integer类静态方法parseInt(String s, int radix)
	 * radix基数,进制
	 * "110",2 含义 前面的数字是二进制的,但是方法parseInt运行结果都是十进制
	 */
	public static void function_1(){
		int i = Integer.parseInt("110", 2);      //表示的是二进制数110，parseInt将二进制数转为十进制数
		System.out.println(i);
	}
	
	/*
	 * Integer类中静态方法 parseInt(String s) 返回基本数据类型
	 * 要求: 字符串必须是数字格式的
	 */
	public static void function(){
		int i = Integer.parseInt("12");
		System.out.println(i/2);
	}
}

```

```java
package cn.itcast.demo1;
/*
 *  Integer类中的其他方法
 *  包括三个方法,和2个静态成员变量
 */
public class IntegerDemo1 {
	public static void main(String[] args) {
		function_1();
	}
	/*
	 * Integer类的3个静态方法
	 * 做进制的转换
	 * 十进制转成二进制  toBinarString(int)
	 * 十进制转成八进制  toOctalString(int)
	 * 十进制转成十六进制  toHexString(int)
	 * 三个方法,返回值都是以String形式出现
	 */
	public static void function_1(){
		System.out.println(Integer.toBinaryString(99));
		System.out.println(Integer.toOctalString(99));
		System.out.println(Integer.toHexString(999));
	}
	
	/*
	 *   Integer类的静态成员变量
	 *   MAX_VALUE
	 *   MIN_VALUE
	 *   输出:Integer的最大范围2147483647      -2147483648
	 */
	public static void function(){
		System.out.println(Integer.MAX_VALUE);
		System.out.println(Integer.MIN_VALUE);
	}
}
```
1.2	基本类型和对象转换
使用int类型与Integer对象转换进行演示，其他基本类型转换方式相同。
	基本数值---->包装对象

Integer i = new Integer(4);//使用构造函数函数
Integer ii = new Integer("4");//构造函数中可以传递一个数字字符串

Integer iii = Integer.valueOf(4);//使用包装类中的valueOf方法
Integer iiii = Integer.valueOf("4");//使用包装类中的valueOf方法

	包装对象---->基本数值

int num = i.intValue();
1.3	自动装箱拆箱
在需要的情况下，基本类型与包装类型可以通用。有些时候我们必须使用引用数据类型时，可以传入基本数据类型。
比如：
	基本类型可以使用运算符直接进行计算，但是引用类型不可以。而基本类型包装类作为引用类型的一种却可以计算，原因在于，Java”偷偷地”自动地进行了对象向基本数据类型的转换。
	相对应的，引用数据类型变量的值必须是new出来的内存空间地址值，而我们可以将一个基本类型的值赋值给一个基本类型包装类的引用。原因同样在于Java又”偷偷地”自动地进行了基本数据类型向对象的转换。
	自动拆箱：对象转成基本数值
	自动装箱：基本数值转成对象
Integer i = 4;//自动装箱。相当于Integer i = Integer.valueOf(4);
i = i + 5;//等号右边：将i对象转成基本数值(自动拆箱) i.intValue() + 5; 加法运算完成后，再次装箱，把基本数值转成对象。

	自动装箱(byte常量池)细节的演示
当数值在byte范围之内时，进行自动装箱，不会新创建对象空间而是使用医来已有的空间。
Integer a = new Integer(3);
Integer b = new Integer(3);
System.out.println(a==b);//false
System.out.println(a.equals(b));//true

System.out.println("---------------------");
Integer x = 127;
Integer y = 127;
//在jdk1.5自动装箱时，如果数值在byte范围之内，不会新创建对象空间而是使用原来已有的空间。
System.out.println(x==y); //true
System.out.println(x.equals(y)); //true


**自动装箱,拆箱的 好处: 基本类型和引用类直接运算
自动装箱和拆箱弊端,可能出现空指针异常**

**//数据在byte范围内,JVM不会从新new对象**
Integer aa = 127; // Integer aa = new Integer(127)++
		Integer bb = 127; // Integer bb = aa;
		System.out.println(aa==bb); **//true**
Integer a = 500;    
		Integer b = 500;
		System.out.println(a==b);**//false**
		System.out.println(a.equals(b));//true

```java
package cn.itcast.demo1;
/*
 *   JDK1.5后出现的特性,自动装箱和自动拆箱
 *   自动装箱: 基本数据类型,直接变成对象
 *   自动拆箱: 对象中的数据变回基本数据类型
 */
public class IntegerDemo2 {
	public static void main(String[] args) {
		function_2();
	}
	/*
	 *  关于自动装箱和拆箱一些题目
	 */
	public static void function_2(){
		Integer i = new Integer(1);
		Integer j = new Integer(1);
		System.out.println(i==j);// false 对象地址
		System.out.println(i.equals(j));// true  继承Object重写equals,比较的对象数据
		
		System.out.println("===================");
		
		Integer a = 500;
		Integer b = 500;
		System.out.println(a==b);//false
		System.out.println(a.equals(b));//true
		
		System.out.println("===================");
		
		
		//数据在byte范围内,JVM不会从新new对象
		Integer aa = 127; // Integer aa = new Integer(127)
		Integer bb = 127; // Integer bb = aa;
		System.out.println(aa==bb); //true
		System.out.println(aa.equals(bb));//true
	}
	
	
	//自动装箱和拆箱弊端,可能出现空指针异常
	public static void function_1(){
	    Integer in =null;	
	    //in = null.intValue()+1
	    in = in + 1;
	    System.out.println(in);
	}
	
	//自动装箱,拆箱的 好处: 基本类型和引用类直接运算
	public static void function(){
		//引用类型 , 引用变量一定指向对象
		//自动装箱, 基本数据类型1, 直接变成了对象
		
		Integer in = 1; // Integer in = new Integer(1)
		//in 是引用类型,不能和基本类型运算, 自动拆箱,引用类型in,转换基本类型
		
		//in+1  ==> in.inValue()+1 = 2    
		// in = 2    自动装箱
		in = in + 1;
		
		System.out.println(in);
		
	}
}
/*
    ArrayList<Integer> ar = new ArrayList<Integer>();
    ar. add(1);
 */
```

```java
package cn.itcast.demo2;

public class SystemDemo {
	public static void main(String[] args) {
		function_4();
	}
	/*
	 * System类方法,复制数组
	 * arraycopy(Object src, int srcPos, Object dest, int destPos, int length)
	 * Object src, 要复制的源数组
	 * int srcPos, 数组源的起始索引
	 * Object dest,复制后的目标数组
	 * int destPos,目标数组起始索引 
	 * int length, 复制几个
	 */
	public static void function_4(){
		int[] src = {11,22,33,44,55,66};
		int[] desc = {77,88,99,0};
		
		System.arraycopy(src, 1, desc, 1, 2);
		for(int i = 0 ;  i < desc.length ; i++){
			System.out.println(desc[i]);
		}
	}
	
	/*
	 *  获取当前操作系统的属性
	 *  static Properties getProperties() 
	 */
	public static void function_3(){
		System.out.println( System.getProperties() );
	}
	
	/*
	 *  JVM在内存中,收取对象的垃圾
	 *  static void gc()
	 */
	public static void function_2(){
		new Person();
		new Person();
		new Person();
		new Person();
		new Person();
		new Person();
		new Person();
		new Person();
		System.gc();
	}
	
	/*
	 *  退出虚拟机,所有程序全停止
	 *  static void exit(0)
	 */
	public static void function_1(){
		while(true){
			System.out.println("hello");
			System.exit(0);
		}
	}
	/*
	 *  获取系统当前毫秒值
	 *  static long currentTimeMillis()
	 *  对程序执行时间测试
	 */
	public static void function(){
		long start = System.currentTimeMillis();
		for(int i = 0 ; i < 10000; i++){
			System.out.println(i);
		}
		long end = System.currentTimeMillis();
		System.out.println(end - start);
	}
}
```

## 第3章 Math类

**3.1	概念**
Math 类是包含用于执行基本数学运算的方法的数学工具类，如初等指数、对数、平方根和三角函数。
类似这样的工具类 ，其所有方法均为静态方法，并且一般不会创建对象。如System类

```java
	abs方法,结果都为正数
double d1 = Math.abs(-5); // d1的值为5
double d2 = Math.abs(5); // d2的值为5
	ceil方法，结果为比参数值大的最小整数的double值
double d1 = Math.ceil(3.3); //d1的值为 4.0
double d2 = Math.ceil(-3.3); //d2的值为 -3.0
double d3 = Math.ceil(5.1); // d3的值为 6.0
	floor方法，结果为比参数值小的最大整数的double值
double d1 = Math.floor(3.3); //d1的值为3.0
double d2 = Math.floor(-3.3); //d2的值为-4.0
double d3 = Math.floor(5.1); //d3的值为 5.0
	max方法，返回两个参数值中较大的值
double d1 = Math.max(3.3, 5.5); //d1的值为5.5
double d2 = Math.max(-3.3, -5.5); //d2的值为-3.3
	min方法，返回两个参数值中较小的值
double d1 = Math.min(3.3, 5.5); //d1的值为3.3
double d2 = Math.max(-3.3, -5.5); //d2的值为-5.5
	pow方法，返回第一个参数的第二个参数次幂的值
double d1 = Math.pow(2.0, 3.0); //d1的值为 8.0
double d2 = Math.pow(3.0, 3.0); //d2的值为27.0
	round方法，返回参数值四舍五入的结果
double d1 = Math.round(5.5); //d1的值为6.0
double d2 = Math.round(5.4); //d2的值为5.0
	random方法，产生一个大于等于0.0且小于1.0的double小数
double d1 = Math.random();
```

## 第4章 Arrays类
.toString .sort .BinarySearch都很实用
**4.1	概念**
此类包含用来操作数组（比如排序和搜索）的各种方法。需要注意，如果指定数组引用为 null，则访问此类中的方法都会抛出空指针异常NullPointerException。
**4.2	常用方法**


```java
	sort方法，用来对指定数组中的元素进行排序（元素值从小到大进行排序）
//源arr数组元素{1,5,9,3,7}, 进行排序后arr数组元素为{1,3,5,7,9}
int[] arr = {1,5,9,3,7};
Arrays.sort( arr );
	toString方法，用来返回指定数组元素内容的字符串形式
int[] arr = {1,5,9,3,7};
String str = Arrays.toString(arr); // str的值为[1, 3, 5, 7, 9]
	binarySearch方法，在指定数组中，查找给定元素值出现的位置。若没有查询到，返回位置为-1。要求该数组必须是个有序的数组。
int[] arr = {1,3,4,5,6};
int index = Arrays.binarySearch(arr, 4); //index的值为2
int index2= Arrasy.binarySearch(arr, 2); //index2的值为-1
4.3	Arrays类的方法练习
	练习一：定义一个方法，接收一个数组，数组中存储10个学生考试分数，该方法要求返回考试分数最低的后三名考试分数。
public static int[] method(double[] arr){
    Arrays.sort(arr); //进行数组元素排序（元素值从小到大进行排序）
    int[] result = new int[3]; //存储后三名考试分数
    System.arraycopy(arr, 0, result, 0, 3);//把arr数组前3个元素复制到result数组中
return result;
}
```

```java
package cn.itcast.demo4;

import java.util.Arrays;

/*
 *  数组的工具类,包含数组的操作
 *  java.util.Arrays
 */
public class ArraysDemo {
	public static void main(String[] args) {
		function_2();
		int[] arr = {56,65,11,98,57,43,16,18,100,200};
		int[] newArray = test(arr);
		System.out.println(Arrays.toString(newArray));
	}
	/*
	 *  定义方法,接收输入,存储的是10个人考试成绩
	 *  将最后三个人的成绩,存储到新的数组中,返回新的数组
	 */
	public static int[] test(int[] arr){
		//对数组排序，默认是从小到大排序
		Arrays.sort(arr);
		//将最后三个成绩存储到新的数组中
		int[] result = new int[3];
		//成绩数组的最后三个元素,复制到新数组中
		System.arraycopy(arr, 0, result, 0, 3);这种方法可以
		for(int i = 0 ;  i < 3 ;i++){
			result[i] = arr[i];
		}
		return result;
	}
	
	/*
	 *  static String toString(数组)
	 *  将数组变成字符串
	 */
	public static void function_2(){
		int[] arr = {5,1,4,6,8,9,0};
		String s = Arrays.toString(arr);
		System.out.println(s);
	}
	
	/*
	 *  static int binarySearch(数组, 被查找的元素)
	 *  数组的二分搜索法
	 *  返回元素在数组中出现的索引
	 *  元素不存在, 返回的是  (-插入点-1)
	 */
	public static void function_1(){
		int[] arr = {1,4,7,9,11,15,18};
	    int index =  Arrays.binarySearch(arr, 10);
	    System.out.println(index);
	}
	
	/*
	 *  static void sort(数组)
	 *  对数组升序排列
	 */
	public static void function(){
		int[] arr = {5,1,4,6,8,9,0};
		Arrays.sort(arr);
		for (int i = 0; i < arr.length; i++) {
			System.out.println(arr[i]);
		}
	}
}
```

## 第5章 大数据运算

**5.1	BigInteger**
  java中long型为最大整数类型,对于超过long型的数据如何去表示呢.在Java的世界中,超过long型的整数已经不能被称为整数了,它们被封装成BigInteger对象.在BigInteger类中,实现四则运算都是方法来实现,并不是采用运算符.
  BigInteger类的构造方法:




```java
构造方法中,采用字符串的形式给出整数
四则运算代码：
/ 
public static void main(String[] args) {
		//大数据封装为BigInteger对象
          BigInteger big1 = new BigInteger("12345678909876543210");
          BigInteger big2 = new BigInteger("98765432101234567890");
          //add实现加法运算
          BigInteger bigAdd = big1.add(big2);   //表示big1-big2  即调用者是被减数，括号内的参数是减数
          //subtract实现减法运算
          BigInteger bigSub = big1.subtract(big2);
          //multiply实现乘法运算
          BigInteger bigMul = big1.multiply(big2);
          //divide实现除法运算
          BigInteger bigDiv = big2.divide(big1);
}
```

**5.2	BigDecimal**

```java
  在程序中执行下列代码,会出现什么问题?
    System.out.println(0.09 + 0.01);
    System.out.println(1.0 - 0.32);
    System.out.println(1.015 * 100);
    System.out.println(1.301 / 100);
```

 double和float类型在运算中很容易丢失精度,造成数据的不准确性,Java提供我们BigDecimal类可以实现浮点数据的高精度运算
   构造方法如下:
    

  建议浮点数据以字符串形式给出,因为参数结果是可以预知的
  实现加法减法乘法代码如下: 

```java
public static void main(String[] args) {
	  //大数据封装为BigDecimal对象
      BigDecimal big1 = new BigDecimal("0.09");
      BigDecimal big2 = new BigDecimal("0.01");
      //add实现加法运算
      BigDecimal bigAdd = big1.add(big2);
      
      BigDecimal big3 = new BigDecimal("1.0");
      BigDecimal big4 = new BigDecimal("0.32");
      //subtract实现减法运算
      BigDecimal bigSub = big3.subtract(big4);
      
      BigDecimal big5 = new BigDecimal("1.105");
      BigDecimal big6 = new BigDecimal("100");
      //multiply实现乘法运算
      BigDecimal bigMul = big5.multiply(big6);
```

  对于浮点数据的除法运算,和整数不同,可能出现无限不循环小数,因此需要对所需要的位数进行保留和选择舍入模式

 BigDecimal高精度计算

```java
package cn.itcast.demo5;

import java.math.BigDecimal;

public class BigDecimalDemo {
	public static void main(String[] args) {
		function_1();
	}
	/*
	 * BigDecimal实现除法运算
	 * divide(BigDecimal divisor, int scale, int roundingMode) 
	 * int scale : 保留几位小数
	 * int roundingMode : 保留模式
	 * 保留模式 阅读API文档
	 *   static int ROUND_UP  向上+1
	 *   static int ROUND_DOWN 直接舍去
	 *   static int ROUND_HALF_UP  >= 0.5 向上+1
	 *   static int ROUND_HALF_DOWN   > 0.5 向上+1 ,否则直接舍去
	 */
	public static void function_1(){
		BigDecimal b1 = new BigDecimal("1.0301");
		BigDecimal b2 = new BigDecimal("100");
		//计算b1/b2的商,调用方法divied
		BigDecimal bigDiv = b1.divide(b2,2,BigDecimal.ROUND_HALF_UP);//0.01301
		System.out.println(bigDiv);
	}
	
	/*
	 *  BigDecimal实现三则运算
	 *  + - *
	 */
	public static void function(){
		BigDecimal b1 =  new BigDecimal("0.09");
		BigDecimal b2 =  new BigDecimal("0.01");
		//计算b1+b2的和,调用方法add
		BigDecimal bigAdd = b1.add(b2);
		System.out.println(bigAdd);
		
		BigDecimal b3 = new BigDecimal("1");
		BigDecimal b4 = new BigDecimal("0.32");
		//计算b3-b2的差,调用方法subtract
		BigDecimal bigSub = b3.subtract(b4);
		System.out.println(bigSub);
		
		BigDecimal b5 = new BigDecimal("1.015");
		BigDecimal b6 = new BigDecimal("100");
		//计算b5*b6的成绩,调用方法 multiply
		BigDecimal bigMul = b5.multiply(b6);
		System.out.println(bigMul);
	}
}


/*
 * 计算结果,未知
 * 原因: 计算机二进制中,表示浮点数不精确造成
 * 超级大型的浮点数据,提供高精度的浮点运算, BigDecimal
System.out.println(0.09 + 0.01);//0.09999999999999999
System.out.println(1.0 - 0.32);//0.6799999999999999
System.out.println(1.015 * 100);//101.49999999999999
System.out.println(1.301 / 100);//0.013009999999999999 
*/
```

## 集合ArrayList  util包下面
集合不存储基本类型，只存储引用类型（即，对象）
```java
package cn.itcast.demo;

import java.util.ArrayList;
/*
 *  集合体系,
 *    目标  集合本身是一个存储的容器:
 *       必须使用集合存储对象
 *       遍历集合,取出对象
 *       集合自己的特性
 */
public class ArrayListDemo {
	public static void main(String[] args) {
		/*
		 *  集合ArrayList,存储int类型数
		 *  集合本身不接受基本类,自动装箱存储
		 */
		ArrayList<Integer> array = new ArrayList<Integer>();
		array.add(11);
		array.add(12);
		array.add(13);
		array.add(14);
		array.add(15);
		for(int i = 0 ; i < array.size() ;i++){
			System.out.println(array.get(i));
		}
		/*
		 *  集合存储自定义的Person类的对象
		 */
		ArrayList<Person> arrayPer = new ArrayList<Person>(); //ArrayList<类名> ArrayList对象名 = new ArrayList<类名>(构造方法)
 - 		arrayPer.add(new Person("a",20));
		arrayPer.add(new Person("b",18));
		arrayPer.add(new Person("c",22));
		for(int i = 0 ; i < arrayPer.size();i++){
			//get(0),取出的对象Person对象
			//打印的是一个对象,必须调用的toString()
			System.out.println(arrayPer.get(i));
		}
	}
}
```

## Collection
查看ArrayList类发现它继承了抽象类AbstractList同时实现接口List，而List接口又继承了Collection接口，Colection接口为最顶层集合接口
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201212213024871.png![在这里插入图片描述](https://img-blog.csdnimg.cn/20201212213048851.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQwMzEwNzEw,size_16,color_FFFFFF,t_70)


![在这里插入图片描述](https://img-blog.csdnimg.cn/20201211203120688.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQwMzEwNzEw,size_16,color_FFFFFF,t_70)
Collection接口概述
	既然Collection接口是集合中的顶层接口，那么他中定义的所有功能子类都可以使用。查阅API中描述的Collection接口。Collection层次结构中的根接口。Collection表示一组对象，这些对象也成为Collection的元素。一些Collection允许有重复的元素，而另一些则不允许。一些Collecion接口是有序的，另一些是无序的。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201207224058187.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQwMzEwNzEw,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201212215021221.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQwMzEwNzEw,size_16,color_FFFFFF,t_70)

```java
package cn.itcast.demo;

import java.util.ArrayList;
import java.util.Collection;

/*
 *  Collection接口中的方法
 *  是集合中所有实现类必须拥有的方法
 *  使用Collection接口的实现类,程序的演示
 *  ArrayList implements List
 *  List extends Collection
 *  方法的执行,都是实现的重写
 */
public class CollectionDemo {
	public static void main(String[] args) {
		function_3();
	}
	/*
	 * Collection接口方法
	 * boolean remove(Object o)移除集合中指定的元素
	 */
	private static void function_3(){
		Collection<String> coll = new ArrayList<String>();
		coll.add("abc");
		coll.add("money");
		coll.add("itcast");
		coll.add("itheima");
		coll.add("money");
		coll.add("123");	
		System.out.println(coll);
		
		boolean b = coll.remove("money");
		System.out.println(b);
		System.out.println(coll);
	}
	
	/*  Collection接口方法
	 *  Object[] toArray() 集合中的元素,转成一个数组中的元素, 集合转成数组
	 *  返回是一个存储对象的数组, 数组存储的数据类型是Object
	 */
	private static void function_2() {
		Collection<String> coll = new ArrayList<String>();
		coll.add("abc");
		coll.add("itcast");
		coll.add("itheima");
		coll.add("money");
		coll.add("123");
		
		Object[] objs = coll.toArray();
		for(int i = 0 ; i < objs.length ; i++){
			System.out.println(objs[i]);
		}
	}
	/*
	 * 学习Java中三种长度表现形式
	 *   数组.length 属性  返回值 int
	 *   字符串.length() 方法,返回值int
	 *   集合.size()方法, 返回值int
	 */
	
	/*
	 * Collection接口方法
	 * boolean contains(Object o) 判断对象是否存在于集合中,对象存在返回true
	 * 方法参数是Object类型
	 */
	private static void function_1() {
		Collection<String> coll = new ArrayList<String>();
		coll.add("abc");
		coll.add("itcast");
		coll.add("itheima");
		coll.add("money");
		coll.add("123");
		
		boolean b = coll.contains("itcast");
		System.out.println(b);
	}


	/*
	 * Collection接口的方法
	 * void clear() 清空集合中的所有元素
	 * 集合容器本身依然存在
	 */
	public static void function(){
		//接口多态的方式调用
		Collection<String> coll = new ArrayList<String>();
		coll.add("abc");
		coll.add("bcd");
		System.out.println(coll);
		
		coll.clear();
		
		System.out.println(coll);
		
	}
}

```

## 迭代器 Iterator
![在这里插入图片描述](https://img-blog.csdnimg.cn/202012122158083.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQwMzEwNzEw,size_16,color_FFFFFF,t_70)

```java
package cn.itcast.demo;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;

/*
 *  集合中的迭代器:
 *    获取集合中元素方式
 *  接口 Iterator : 两个抽象方法
 *     boolean hasNext() 判断集合中还有没有可以被取出的元素,如果有返回true
 *     next() 取出集合中的下一个元素
 *     
 *  Iterator接口,找实现类.
 *    Collection接口定义方法 
 *       Iterator  iterator()
 *    ArrayList 重写方法 iterator(),返回了Iterator接口的实现类的对象
 *    使用ArrayList集合的对象
 *     Iterator it = array.iterator(),运行结果就是Iterator接口的实现类的对象
 *     it是接口的实现类对象,调用方法 hasNext 和 next 集合元素迭代
 */
public class IteratorDemo {
	public static void main(String[] args) {
		Collection<String> coll = new ArrayList<String>();
		coll.add("abc1");
		coll.add("abc2");
		coll.add("abc3");
		coll.add("abc4");
		//迭代器,对集合ArrayList中的元素进行取出
		
		//调用集合的方法iterator()获取出,Iterator接口的实现类的对象
		Iterator<String> it = coll.iterator();
		//接口实现类对象,调用方法hasNext()判断集合中是否有元素
		//boolean b = it.hasNext();
		//System.out.println(b);
		//接口的实现类对象,调用方法next()取出集合中的元素
		//String s = it.next();
		//System.out.println(s);
		
		//迭代是反复内容,使用循环实现,循环的条件,集合中没元素, hasNext()返回了false
		while(it.hasNext()){
			String s = it.next();
			System.out.println(s);
		}
		
		/*for (Iterator<String> it2 = coll.iterator(); it2.hasNext();  ) {
			System.out.println(it2.next());
		}*/
		
	}
}

```
## 增强for循环 ForEachDemo
JDK1.5以后的一个高级for循环，专门用来遍历数组和集合的。它的内部原理其实是个Iterator迭代器，所以在遍历的过程中，不能对集合中的元素进行增删操作，不能进行排序等操作，因为没有索引，不能操作容器里面的元素。
```java
package cn.itcast.demo2;
import java.util.ArrayList;

/*
 *  JDK1.5新特性,增强for循环
 *  JDK1.5版本后,出现新的接口 java.lang.Iterable
 *    Collection开是继承Iterable
 *    Iterable作用,实现增强for循环
 *    
 *    格式:
 *      for( 数据类型  变量名 : 数组或者集合 ){
 *         sop(变量);
 *      }
 */
import cn.itcast.demo.Person;
public class ForEachDemo {
	public static void main(String[] args) {
		function_2();
	}
	/*
	 *  增强for循环遍历集合
	 *  存储自定义Person类型
	 */
	public static void function_2(){
		ArrayList<Person> array = new ArrayList<Person>();
		array.add(new Person("a",20));
		array.add(new Person("b",10));
		for(Person p : array){
			System.out.println(p);
		}
	}
	
	
	public static void function_1(){
		//for对于对象数组遍历的时候,能否调用对象的方法呢
		String[] str = {"abc","itcast","cn"};
		for(String s : str){     //for(类型 变量名： 数组名)
			System.out.println(s.length());
		}
	}
	
	/*
	 *  实现for循环,遍历数组
	 *  好处: 代码少了,方便对容器遍历
	 *  弊端: 没有索引,不能操作容器里面的元素
	 */
	public static void function(){
		int[] arr = {3,1,9,0};
		for(int i : arr){
			System.out.println(i+1);
		}
		System.out.println(arr[0]);
	}
}
```


## 泛型
用于解决类型转换异常的安全问题
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201212224851975.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQwMzEwNzEw,size_16,color_FFFFFF,t_70)

**泛型: 指明了集合中存储数据的类型  <数据类型>**
	Collection<String> coll = new ArrayList<String>();
```java
package cn.itcast.demo3;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
/*
 * JDK1.5 出现新的安全机制,保证程序的安全性
 *   
 */

public class GenericDemo {
	public static void main(String[] args) {
		function();
	}
	
	public static void function(){
		Collection<String> coll = new ArrayList<String>();   //这句话就是泛型，指明col1存储的对象类型是String
		coll.add("abc");
		coll.add("rtyg");
		coll.add("43rt5yhju");
//		coll.add(1);
		
		Iterator<String> it = coll.iterator();
		while(it.hasNext()){
			String s = it.next();
			System.out.println(s.length());
		}
	}
}
```

## 1.2 List接口中常用的方法

	增加元素方法
		add(Object e)：向集合末尾处，添加指定的元素 
		add(int index, Object e)：向集合指定索引处，添加指定的元素，原有元素依次后移
	删除元素删除
		remove(Object e)：将指定元素对象，从集合中删除，返回值为被删除的元素
		remove(int index)：将指定索引处的元素，从集合中删除，返回值为被删除的元素
	替换元素方法
		set(int index, Object e)：将指定索引处的元素，替换成指定的元素，返回值为替换前的元素
	查询元素方法
		get(int index)：获取指定索引处的元素，并返回该元素

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210115092944692.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQwMzEwNzEw,size_16,color_FFFFFF,t_70)

```java
package cn.itcast.demo;

import java.util.ArrayList;
import java.util.List;

/*
 *  List接口派系, 继承Collection接口
 *    下面有很多实现类
 *  List接口特点: 有序,索引,可以重复元素
 *    实现类, ArrayList, LinkedList
 *    
 *  List接口中的抽象方法,有一部分方法和他的父接口Collection是一样
 *  List接口的自己特有的方法, 带有索引的功能
 */
public class ListDemo {
	public static void main(String[] args) {
		function_2();
	}
	/*
	 *  E set(int index, E)
	 *  修改指定索引上的元素
	 *  返回被修改之前的元素
	 */
	public static void function_2(){
		List<Integer> list = new ArrayList<Integer>();
		list.add(1);
		list.add(2);
		list.add(3);
		list.add(4);
		
		Integer i = list.set(0, 5);  将0索引上的值改为5
		System.out.println(i);
		System.out.println(list);
	}
	
	/*
	 *  E remove(int index)
	 *  移除指定索引上的元素
	 *  返回被删除之前的元素
	 */
	public static void function_1(){
		List<Double> list = new ArrayList<Double>();
		list.add(1.1);
		list.add(1.2);
		list.add(1.3);
		list.add(1.4);
		
		Double d = list.remove(0);  删除的是0索引上的1.1  remove方法返回被删除的值1.1
		System.out.println(d);
		System.out.println(list);
	}
	
	/*
	 *  add(int index, E)
	 *  将元素插入到列表的指定索引上
	 *  带有索引的操作,防止越界问题
	 *  java.lang.IndexOutOfBoundsException
	 *     ArrayIndexOutOfBoundsException
	 *     StringIndexOutOfBoundsException
	 */
	public static void function(){
		List<String> list = new ArrayList<String>();
		list.add("abc1");
		list.add("abc2");
		list.add("abc3");
		list.add("abc4");
		System.out.println(list);
		
		list.add(1, "itcast");
		System.out.println(list);
	}
}
```

## 1.2.1 Iterator的并发修改异常
简单描述：  我一边遍历你一边加东西  就会报错
**迭代器工作的时候，不能修改集合的长度**
/*
 *  迭代器的并发修改异常 java.util.ConcurrentModificationException
 *  就是在遍历的过程中,使用了集合方法修改了集合的长度,不允许的
 */
运行上述代码发生了错误 java.util.ConcurrentModificationException 这是什么原因呢？
在迭代过程中，使用了集合的方法对元素进行操作。导致迭代器并不知道集合中的变化，容易引发数据的不确定性。
并发修改异常解决办法：在迭代时，不要使用集合的方法操作元素。
那么想要在迭代时对元素操作咋办？通过ListIterator迭代器操作元素是可以的，ListIterator的出现，解决了使用Iterator迭代过程中可能会发生的错误情况。

```java
package cn.itcast.demo;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/*
 *  迭代器的并发修改异常 java.util.ConcurrentModificationException
 *  就是在遍历的过程中,使用了集合方法修改了集合的长度,不允许的
 */
public class ListDemo1 {
	public static void main(String[] args) {
		List<String> list = new ArrayList<String>();
		list.add("abc1");
		list.add("abc2");
		list.add("abc3");
		list.add("abc4");
		
		//对集合使用迭代器进行获取,获取时候判断集合中是否存在 "abc3"对象
		//如果有,添加一个元素 "ABC3"
		Iterator<String> it = list.iterator();
		while(it.hasNext()){
			String s = it.next();
			//对获取出的元素s,进行判断,是不是有"abc3"
			if(s.equals("abc3")){  不能用s == "abc3"来判断   字符串
				list.add("ABC3");
			}
			System.out.println(s);
		}
	}
}

```


```java
各个数据结构的特点：
**	堆栈**，采用该结构的集合，对元素的存取有如下的特点：
			先进后出（即，存进去的元素，要在后它后面的元素依次取出后，才能取出该元素）。例如，子弹压进弹夹，先压进去的子弹在下面，后压进去的子弹在上面，当开枪时，先弹出上面的子弹，然后才能弹出下面的子弹。
			栈的入口、出口的都是栈的顶端位置
			压栈：就是存元素。即，把元素存储到栈的顶端位置，栈中已有元素依次向栈底方向移动一个位置。
			弹栈：就是取元素。即，把栈的顶端位置元素取出，栈中已有元素依次向栈顶方向移动一个位置。
**	队列**，采用该结构的集合，对元素的存取有如下的特点：
			先进先出（即，存进去的元素，要在后它前面的元素依次取出后，才能取出该元素）。例如，安检。排成一列，每个人依次检查，只有前面的人全部检查完毕后，才能排到当前的人进行检查。
			队列的入口、出口各占一侧。例如，下图中的左侧为入口，右侧为出口。
**	数组**，采用该结构的集合，对元素的存取有如下的特点：
			*查找元素快*：通过索引，可以快速访问指定位置的元素
			*增删元素慢*：
				指定索引位置增加元素：需要创建一个新数组，将指定新元素存储在指定索引位置，再把原数组元素根据索引，复制到新数组对应索引的位置。如下图
				指定索引位置删除元素：需要创建一个新数组，把原数组元素根据索引，复制到新数组对应索引的位置，原数组中指定索引位置元素不复制到新数组中
**	链表**，采用该结构的集合，对元素的存取有如下的特点：
			多个节点之间，通过地址进行连接。例如，多个人手拉手，每个人使用自己的右手拉住下个人的左手，依次类推，这样多个人就连在一起了。
			*查找元素慢*：想查找某个元素，需要通过连接的节点，依次向后查找指定元素
			*增删元素快*：
				增加元素：操作如左图，只需要修改连接下个元素的地址即可。
				删除元素：操作如右图，只需要修改连接下个元素的地址即可。
```

ArrayList是一个数组列表
元素增删慢 查找快
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210115103445121.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQwMzEwNzEw,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021011510345255.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQwMzEwNzEw,size_16,color_FFFFFF,t_70)

## 1.5 LinkedList集合
LinkedList集合数据存储的结构是链表结构。方便元素添加、删除的集合。实际开发中对一个集合元素的添加与删除经常涉及到首尾操作，而LinkedList提供了大量首尾操作的方法。如下图
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210115104037449.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQwMzEwNzEw,size_16,color_FFFFFF,t_70)
LinkedList是List的子类，List中的方法LinkedList都是可以使用，这里就不做详细介绍，我们只需要了解LinkedList的特有方法即可。在开发时，LinkedList集合也可以作为堆栈，队列的结构使用。

```java
package cn.itcast.demo;

import java.util.LinkedList;

/*
 *  LinkedList 链表集合的特有功能
 *    自身特点: 链表底层实现,查询慢,增删快
 *  
 *  子类的特有功能,不能多态调用
 */
public class LinkedListDemo {
	public static void main(String[] args) {
		function_3();
	}
	/*
	 *  E removeFirst() 移除并返回链表的开头
	 *  E removeLast() 移除并返回链表的结尾
	 */
	public static void function_3(){
		LinkedList<String> link = new LinkedList<String>();
		link.add("1");
		link.add("2");
		link.add("3");
		link.add("4");
		
		String first = link.removeFirst();
		String last = link.removeLast();
		System.out.println(first);
		System.out.println(last);
	
		System.out.println(link);
	}
	
	/*
	 * E getFirst() 获取链表的开头
	 * E getLast() 获取链表的结尾
	 */
	public static void function_2(){
		LinkedList<String> link = new LinkedList<String>();
		link.add("1");
		link.add("2");
		link.add("3");
		link.add("4");
	    //link.clear();清空操作
		if(!link.isEmpty()){
			String first = link.getFirst();
			String last = link.getLast();
			System.out.println(first);
			System.out.println(last);
		}
	}
	
	public static void function_1(){
		LinkedList<String> link = new LinkedList<String>();
		link.addLast("a");
		link.addLast("b");
		link.addLast("c");
		link.addLast("d");
		
		link.addFirst("1");
		link.addFirst("2");
		link.addFirst("3");
		System.out.println(link);
	}
	
	/*
	 *  addFirst(E) 添加到链表的开头
	 *  addLast(E) 添加到链表的结尾
	 */
	public static void function(){
		LinkedList<String> link = new LinkedList<String>();
		
		link.addLast("heima");
		
		link.add("abc");
		link.add("bcd");
		
		link.addFirst("itcast");
		System.out.println(link);
		
		
	}
}

```

## 1.6 Vector集合
Vector集合数据存储的结构是数组结构，为JDK中最早提供的集合。Vector中提供了一个独特的取出方式，就是枚举Enumeration，它其实就是早期的迭代器。此接口Enumeration的功能与 Iterator 接口的功能是类似的。Vector集合已被ArrayList替代。枚举Enumeration已被迭代器Iterator替代。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210115105807211.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQwMzEwNzEw,size_16,color_FFFFFF,t_70)

**1、println和printf以及print区别**
print和prinln的区别     后者自带换行功能
printf--函数，把文字格式化以后输出，直接调用系统调用进行IO的，他是非缓冲的。
**2、Integer和int的区别**

1、Integer是int提供的封装类，而int是Java的基本数据类型；
2、Integer默认值是null，而int默认值是0；
3、声明为Integer的变量需要实例化，而声明为int的变量不需要实例化；
4、Integer是对象，用一个引用指向这个对象，而int是基本类型，直接存储数值。


```java
通过数组转为链表，以及通过键盘输入链表  考试可能第一种偏多

合并两个链表代码
package leetcode1230;
import java.util.*;

public class ListNodeMOBAN {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		Scanner in = new Scanner(System.in);
		String line = in.nextLine();
		Scanner in2 = new Scanner(line);
		//数组转链表
		ListNode l5 = new ListNode(0);
		ListNode cur3 = l5;
		int[] arr = {1,2,3,4,5};
		for(int i=0; i<arr.length; i++){
			l5.next = new ListNode(arr[i]);
			l5 = l5.next;
		}
		
		//链表1
		ListNode l4 = new ListNode(0);
		ListNode cur2 = l4;
		while(in2.hasNextInt()){
	     	l4.next = new ListNode(in2.nextInt());
	     	l4 = l4.next;
	 	}
		
		
		ListNode ret2 = mergeTwoLists(cur2.next,cur3.next);
		System.out.println("");
		System.out.println("l1和l4");
		while(ret2!=null){
			System.out.printf("%d  ",ret2.val);
			ret2 = ret2.next;
		}
	}
	public static ListNode mergeKLists(ListNode[] lists) {
        ListNode ans = null;
        for (int i = 0; i < lists.length; ++i) {
            ans = mergeTwoLists(ans, lists[i]);
        }
        return ans;
    }

    public static ListNode mergeTwoLists(ListNode a, ListNode b){
        if (a == null || b == null) {
            return a != null ? a : b;
        }
        ListNode head = new ListNode(0);
        ListNode tail = head, aPtr = a, bPtr = b;
        while (aPtr != null && bPtr != null) {
            if (aPtr.val < bPtr.val) {
                tail.next = aPtr;
                aPtr = aPtr.next;
            } else {
                tail.next = bPtr;
                bPtr = bPtr.next;
            }
            tail = tail.next;
        }
        tail.next = (aPtr != null ? aPtr : bPtr);
        return head.next;
    }
	
}
```

