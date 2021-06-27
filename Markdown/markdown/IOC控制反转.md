## IOC控制反转

****

**1、什么是IOC（控制反转）**

 a）把对象创建和对象之间的调用过程，交给Spring进行管理

 b）使用IOC目的：为了降低耦合度

 **2、IOC底层**

 a）xml解析、工厂模式、反射

 **3、Spring提供的IOC容器实现的两种方式（两个接口）**

**IOC（接口）**

1、IOC思想基于IOC容器完成，IOC容器底层就是对象工厂

 a）BeanFactory接口：IOC容器基本实现，是Spring内部接口的使用接口，不提供给开发人员进行使用（加载配置文件时候不会创建对象，在获取对象时才会创建对象。）

 b）ApplicationContext接口：BeanFactory接口的子接口，提供更多更强大的功能，提供给开发人员使用（加载配置文件时候就会把在配置文件对象进行创建）推荐使用！

**4、ApplicationContext接口的实现类（具体根据API文档查看☺）**





解耦：

原始方式：

工厂模式：

**5、IOC过程**

1、xml配置文件。配置创建的对象

```java
<bean id="user" class="com.sdl.spring5.User"> </bean>
```

2、有service类和dao类，创建工厂类

```java
class UserFactory{
    public static UserDao getDao(){
        String classValue = class属性值; //根据xml解析得到
        Class clazz = Class.forName(classValue);//通过反射创建对象
        return (UserDao)clazz.newInstance(); //强转换return
        
    }
}
这种做法只需要修改xml文件，其他部分不需要修改
```

## IOC容器-Bean管理

1、IOC操作Bean管理

 a）Bean管理就是两个操作：（1）Spring创建对象；（2）Spring注入属性

2、基于XML配置文件创建对象

a、在spring配置文件中，使用bean标签，标签里面添加对应属性，就可以实现对象创建

b、常用属性

（1）**ID属性：**唯一标识

（2）**class属性：**类全路径（包含类路径）

（3）name属性：可以加特殊符号，功能和ID类似，基本不用了

c、创建对象的时候，默认也是执行无参数构造方法完成对象创建

```java
<!--1 配置User对象创建-->
<bean id="user" class="com.atguigu.spring5.User"></bean>
```

 3、基于XML方式注入属性（DI：依赖注入（注入属性））

 （1）DI：依赖注入，就是注入属性

​			第一种注入方式：使用set方法进行注入

```java
            public class Book {
                private String bname;
                public void setBname(String bname) {
                    this.bname = bname;
                }
                public static void main(String[] args) {
                    Book book = new Book();
                    book.setBname("abc");
                }
            }
```

​			第二种注入方式：使用有参数构造进行注入

```java
			public Book(String bname){
                this.bname = bname;
            }
```



a）set方式注入

```java
//（1）传统方式： 创建类，定义属性和对应的set方法
public class Book {
        //创建属性
        private String bname;

        //创建属性对应的set方法
        public void setBname(String bname) {
            this.bname = bname;
        }
   }
```

```java
<!--（2）spring方式： set方法注入属性-->
<bean id="book" class="com.atguigu.spring5.Book">
    <!--使用property完成属性注入
        name：类里面属性名称
        value：向属性注入的值
    -->
    <property name="bname" value="Hello"></property>
    <property name="bauthor" value="World"></property>
</bean>
```

 b）有参构造函数注入

在spring配置文件中进行配置

```java
//（1）传统方式：创建类，构建有参函数
public class Orders {
    //属性
    private String oname;
    private String address;
    //有参数构造
    public Orders(String oname,String address) {
        this.oname = oname;
        this.address = address;
    }
  }

```

```java
<!--（2）spring方式：有参数构造注入属性-->
<bean id="orders" class="com.atguigu.spring5.Orders">
    <constructor-arg name="oname" value="Hello"></constructor-arg>
    <constructor-arg name="address" value="China！"></constructor-arg>
</bean>
注释：还要对应修改一个testAdd()
```

 c）p名称空间注入（了解即可）

（1）使用p名称空间注入，可以简化基于xml配置方式

第一步：添加p名称空间在配置文件中

```java
xmlns:p="http://www.springframework.org/schema/p"
```

第二步：进行属性注入，在bean标签里面进行操作

```java
<bean id="book" class="com.atguigu.spring5.Book" p:bname="very" p:bauthor="good">
```

```java
<!--1、添加p名称空间在配置文件头部-->
<?xml version="1.0" encoding="UTF-8"?>
<beans xmlns="http://www.springframework.org/schema/beans"
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xmlns:p="http://www.springframework.org/schema/p"		<!--在这里添加一行p-->

<!--2、在bean标签进行属性注入（算是set方式注入的简化操作）-->
    <bean id="book" class="com.atguigu.spring5.Book" p:bname="九阳神功" p:bauthor="无名氏">
    </bean>
```

 4、注入空值和特殊符号

属性值包含特殊符号

（1）null值

```java
<property name="address"> <null/> </property>
```

（2）错误写法

```java
<property name="address"> value = "<<南京>>" </property>   这样写报错啦
```

方式1：利用转义字符

```java
		<property name="address">
            value = "&lt;&lt;南京;&gt;&gt"
        </property>
```

**方式2：常用记住**

```java
		<property name="address">
            <value><![CDATA[<<南京>>]]></value>
        </property>
```

    <bean id="book" class="com.atguigu.spring5.Book">
        <!--（1）null值-->
        <property name="address">
            <null/><!--属性里边添加一个null标签-->
        </property>
        <!--（2）特殊符号赋值-->
     	  <!--属性值包含特殊符号
           a 把<>进行转义 &lt; &gt;
           b 把带特殊符号内容写到CDATA
          -->
            <property name="address">
                <value><![CDATA[<<南京>>]]></value>
            </property>
    </bean>
**5、注入属性-外部bean**

 a）创建两个类service和dao类

 b）在service调用dao里面的方法

 c）在spring的配置文件中进行配置

注意：重写是Override大写开头

```JAVA

public class UserService {//service类

    //创建UserDao类型属性，生成set方法
    private UserDao userDao;
    public void setUserDao(UserDao userDao) {
        this.userDao = userDao;
    }

    public void add() {
        System.out.println("service add...............");
        userDao.update();//调用dao方法
    }
}

public class UserDaoImpl implements UserDao {//dao类

    @Override
    public void update() {
        System.out.println("dao update...........");
    }
}
```

 b）在spring配置文件中进行配置

​			name属性：类里面属性名称
​            ref属性：创建userDao对象bean标签id值

````JAVA
<?xml version="1.0" encoding="UTF-8"?>
<beans xmlns="http://www.springframework.org/schema/beans"
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xmlns:p="http://www.springframework.org/schema/p"
       xsi:schemaLocation="http://www.springframework.org/schema/beans http://www.springframework.org/schema/beans/spring-beans.xsd">

    <!--1 service和dao对象创建-->
    <bean id="userService" class="com.sdl.demo_test.service.UserService">
        <!--注入userDao对象
            name属性：类里面属性名称
            ref属性：创建userDao对象bean标签id值
        -->
        <property name="userDao" ref="userDaoImpl"></property>
    </bean>
    <bean id="userDaoImpl" class="com.sdl.demo_test.dao.UserDaoImpl"></bean>
</beans>
````

 **6、基于XML方式注入内部bean和级联赋值**

（1）一对多关系：部门和员工

一个部门有多个员工，一个员工属于多个部门，部门是一，员工是多

（2）在实体类之间表示一对多关系，员工表示所属部门，适用对象类型属性进行表示

**a）注入属性-内部bean**

```JAVA
//部门类
public class Dept {
    private String dname;
    public void setDname(String dname) {
        this.dname = dname;
    }
}

//员工类
public class Emp {
    private String ename;
    private String gender;
    //员工属于某一个部门，使用对象形式表示
    private Dept dept;
    
    public void setDept(Dept dept) {
        this.dept = dept;
    }
    public void setEname(String ename) {
        this.ename = ename;
    }
    public void setGender(String gender) {
        this.gender = gender;
    }
}
```



（3）在spring配置文件中配置

```java
<!--内部bean-->
    <bean id="emp" class="com.atguigu.spring5.bean.Emp">
        <!--设置两个普通属性-->
        <property name="ename" value="Andy"></property>
        <property name="gender" value="女"></property>
        <!--设置对象类型属性-->
        <property name="dept">
            <bean id="dept" class="com.atguigu.spring5.bean.Dept"><!--内部bean赋值-->
                <property name="dname" value="宣传部门"></property>
            </bean>
        </property>
    </bean>
```

**b）注入属性-级联赋值**

重要：级联bin  

1、一个Bean，通过dept.dname来级联

2、两个Bean，第一个bean设置emp中的所有属性，第二个bean设置级联的dept内部的属性

```java
<property name="dept" ref="dept"></property>
        <property name="dept.dname" value="技术部门"></property>
```

```java
<!--方式一：级联赋值-->
    <bean id="emp" class="com.atguigu.spring5.bean.Emp">
        <!--设置两个普通属性-->
        <property name="ename" value="Andy"></property>
        <property name="gender" value="女"></property>
        <!--级联赋值-->
        <property name="dept" ref="dept"></property>
    </bean>
    <bean id="dept" class="com.atguigu.spring5.bean.Dept">
        <property name="dname" value="公关部门"></property>
    </bean>
```

```java
 //方式二：生成dept的get方法（get方法必须有！！）
    public Dept getDept() {
        return dept;
    }
```

```java
<!--级联赋值-->
    <bean id="emp" class="com.atguigu.spring5.bean.Emp">
        <!--设置两个普通属性-->
        <property name="ename" value="jams"></property>
        <property name="gender" value="男"></property>
        <!--级联赋值-->
        <property name="dept" ref="dept"></property>
        <property name="dept.dname" value="技术部门"></property>
    </bean>
    <bean id="dept" class="com.atguigu.spring5.bean.Dept">
    </bean>
```

**7、IOC 操作 Bean 管理——xml 注入集合属性**

注入数组类型属性 2、注入 List 集合类型属性 3、注入 Map 集合类型属性

```java
//（1）创建类，定义数组、list、map、set 类型属性，生成对应 set 方法
public class Stu {
    //1 数组类型属性
    private String[] courses;
    //2 list集合类型属性
    private List<String> list;
    //3 map集合类型属性
    private Map<String,String> maps;
    //4 set集合类型属性
    private Set<String> sets;
    
    public void setSets(Set<String> sets) {
        this.sets = sets;
    }
    public void setCourses(String[] courses) {
        this.courses = courses;
    }
    public void setList(List<String> list) {
        this.list = list;
    }
    public void setMaps(Map<String, String> maps) {
        this.maps = maps;
    }
```

**1、注入数组类型属性**

**2、注入list集合类型属性**

**3、注入map集合类型属性**

```java
package com.sdl.spring.collectiontype;
import java.util.*;
public class Stu {
	//1 数组类型属性
	private String[] courses;

	//2 list集合类型属性
	private List<String> list;
	
	//3 map集合类型属性
	private Map<String,String> maps;
	
	//4 Set集合类型属性
	private Set<String> sets;
	
	public void setCourses(String[] courses) {
		this.courses = courses;
	}


	public void setMaps(Map<String,String> maps) {
		this.maps = maps;
	}


	public void setList(List<String> list) {
		this.list = list;
	}


	public void setSets(Set<String> sets) {
		this.sets = sets;
	}
	
	public void test() {
		System.out.println(Arrays.toString(courses));
		System.out.println(this.list);
		System.out.println(this.maps);
		System.out.println(this.sets);
	}
}

```



（2）在Spring配置文件进行配置

```java
<?xml version="1.0" encoding="UTF-8"?>
<beans xmlns="http://www.springframework.org/schema/beans"
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xmlns:p="http://www.springframework.org/schema/p"
       xsi:schemaLocation="http://www.springframework.org/schema/beans http://www.springframework.org/schema/beans/spring-beans.xsd">

    <!--1 集合类型属性的注入-->
   	<bean id="stu" class="com.sdl.spring.collectiontype.Stu">
   		<!-- 数组类型注入 -->
   		<property name = "courses">
   			<array>
   				<value>java课程</value>
   				<value>数据库课程</value>
   			</array>
   		</property>
   		
   		<!-- list集合属性注入 -->
   		<property name = "list">
   			<list>
   				<value>张三</value>
   				<value>小三</value>
   			</list>
   		</property>
   		
   		<!-- map类型属性注入 -->
   		<property name = "maps">
   			<map>   <!-- 这里的map是系统中就有的所以是map而不是maps -->
   				<entry key = "JAVA" value="java"></entry>
   				<entry key = "PHP" value="php"></entry>
   			</map>
   		</property>
   		
   		<!-- set类型属性注入 -->
   		<property name = "sets">
   			<set>
   				<value>MySQL</value>
   				<value>Redis</value>
   			</set>
   		</property>
   	</bean>
   	
</beans>
```

**4、在集合里面设置对象类型值**

```java
<?xml version="1.0" encoding="UTF-8"?>
<beans xmlns="http://www.springframework.org/schema/beans"
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xmlns:p="http://www.springframework.org/schema/p"
       xsi:schemaLocation="http://www.springframework.org/schema/beans http://www.springframework.org/schema/beans/spring-beans.xsd">

    <!--1 集合类型属性的注入-->
   	<bean id="stu" class="com.sdl.spring.collectiontype.Stu">
   		<!-- 数组类型注入 -->
   		<property name = "courses">
   			<array>
   				<value>java课程</value>
   				<value>数据库课程</value>
   			</array>
   		</property>
   		
   		<!-- list集合属性注入 -->
   		<property name = "list">
   			<list>
   				<value>张三</value>
   				<value>小三</value>
   			</list>
   		</property>
   		
   		<!-- map类型属性注入 -->
   		<property name = "maps">
   			<map>   <!-- 这里的map是系统中就有的所以是map而不是maps -->
   				<entry key = "JAVA" value="java"></entry>
   				<entry key = "PHP" value="php"></entry>
   			</map>
   		</property>
   		
   		<!-- set类型属性注入 -->
   		<property name = "sets">
   			<set>
   				<value>MySQL</value>
   				<value>Redis</value>
   			</set>
   		</property>
   		
   		<!-- 注入List<Course> -->
   		<property name = "courseList">
   			<list>
   				<ref bean = "course1"></ref>
   				<ref bean = "course2"></ref>
   			</list>
   		</property>
   	</bean>
	<!-- 创建多个course对象 -->
	<bean id = "course1" class="com.sdl.collectiontype.Course">
		<property name = "cname" value = "Spring5框架"></property>
	</bean>
	
	<bean id = "course2" class="com.sdl.collectiontype.Course">
		<property name = "cname" value = "MyBatis框架"></property>
	</bean>
</beans>
```



**5、把集合注入部分提取出来**

（1）在spring配置文件中引入名称空间  util

```java
<?xml version="1.0" encoding="UTF-8"?>
<beans xmlns="http://www.springframework.org/schema/beans"
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xmlns:p="http://www.springframework.org/schema/p"
       xmlns:util="http://www.springframework.org/schema/util"
       xsi:schemaLocation="http://www.springframework.org/schema/beans http://www.springframework.org/schema/beans/spring-beans.xsd
       http://www.springframework.org/schema/util http://www.springframework.org/schema/util/spring-util.xsd">
```
（2）使用util标签完成list集合注入提取

```java
<!-- 提取list集合类型属性注入 -->
   	<util:list id = "bookList">
   		<value>易筋经</value>
   		<value>九阴真经</value>
   		<value>九阳神功</value>
   	</util:list>
```

### IOC操作Bean管理（FactoryBean）

 **1、Spring 有两种类型 bean，一种普通 bean，另外一种工厂 bean（FactoryBean）**

 **2、普通 bean：在配置文件中*定义 bean 类型就是返回类型*****

 **3、工厂 bean：在配置文件*定义 bean 类型可以和返回类型不一样*** 

​			第一步 创建类，让这个类作为工厂 bean，实现接口 FactoryBean 

​			第二步 实现接口里面的方法，在实现的方法中定义返回的 bean 类型

```java
package com.sdl.spring.factorybean;

import org.springframework.beans.factory.FactoryBean;

import com.sdl.spring.collectiontype.Course;

public class MyBean implements FactoryBean<Course>{

	//定义返回bean
	@Override
	public Course getObject() throws Exception {
		// TODO Auto-generated method stub
		Course course = new Course();
		course.setCname("abc");
		return course;
	}

	@Override
	public Class getObjectType() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public boolean isSingleton() {
		return false;
	}
}

```

```java
@Test
	public void test3() {
		ApplicationContext context = 
				new ClassPathXmlApplicationContext("bean3.xml");
		Course course = context.getBean("myBean",Course.class);
		System.out.println(course);
		
	}
```

```java
<bean id = "myBean" class = "com.sdl.spring.factorybean.MyBean" scope="prototype">
   	</bean>
```

### IOC操作Bean管理（bean的作用域）

**1、Spring，设置创建bean实例是单实例还是多实例**

**2、在spring里面，默认情况下，bean是单实例**

**3、如何设置单实例还是多实例**

（1）在 spring 配置文件 bean 标签里面有属性（scope）用于设置单实例还是多实例

（2）scope 属性值 
    第一个值 默认值，singleton，表示是单实例对象 
    第二个值 prototype，表示是多实例对象
  (3)  singleton和prototype区别
    **a**. singleton单实例,prototype多实例
    **b**. 设置scope值是singleton时候,**加载spring配置文件的时候就会创建单实例对象**
        设置scope值是prototype的时候,不是在加载spring配值文件时候创建对象,在**调用getBean方法时候创建多实例对象**

```java
<bean id="book" class="com.sdl.spring.collectiontype.Book"><!--设置为单实例-->
        <property name="list" ref="bookList"></property>
</bean>
```

```java
com.sdl.spring.collectiontype.Book@dd0c991
com.sdl.spring.collectiontype.Book@dd0c991
地址完全相同，说明是单实例
```

```java
<bean id="book" class="com.sdl.spring.collectiontype.Book" scope="prototype"><!--设置为多实例-->
        <property name="list" ref="bookList"></property>
</bean>
```

```java
com.sdl.spring.collectiontype.Book@5f16132a
com.sdl.spring.collectiontype.Book@69fb6037
两个book地址不同，说明是多实例
```



### IOC操作Bean管理（bean生命周期）

****

**1、生命周期**

​	(1) 从对象创建到对象销毁的过程

**2、bean生命周期**

​	(1) 通过构造器创建bean实例(无参构造)

​	(2) 为bean的属性设置值和对其他bean引用(调用set方法)

​    (3) 调用bean的初始化的方法(需要进行配置)

​    (4) bean可以使用了(对象获取到了)

​	(5) 当容器关闭时候,调用bean的销毁方法(需要进行配置销毁的方法)

**3、演示bean生命周期**

Order类

```java
package com.sdl.spring.bean;

public class Orders {
	private String oname;
	
	public Orders() {
		System.out.println("1 通过构造器创建bean实例(无参构造)");
	}
	public void setOname(String oname) {
		this.oname = oname;
		System.out.println("2 为bean的属性设置值和对其他bean引用(调用set方法)");
	}
	//3 创建执行的初始化方法
	public void initMethod() {
		System.out.println("3 执行初始化的方法");
	}
	//4 创建执行的销毁方法
	public void destroyMethod() {
		System.out.println("5 执行销毁的方法");
	}
}

```

TestDemo

```java
	@Test
	public void testBean3() {
//		ApplicationContext context = 
//				new ClassPathXmlApplicationContext("bean4.xml");
		ClassPathXmlApplicationContext context = 
				new ClassPathXmlApplicationContext("bean4.xml");
		Orders orders = context.getBean("orders",Orders.class);
		System.out.println("4 获取创建bean实例对象");
		System.out.println(orders);
		
		
		//手动销毁
		//((ClassPathXmlApplicationContext) context).close();
		context.close();
	}
```

bean4.xml

```java
	<bean id = "orders" class = "com.sdl.spring.bean.Orders" init-method = "initMethod" destroy-method = "destroyMethod">
   		<property name="oname" value = "手机"></property>
   	</bean>
```



```java
Output
1 通过构造器创建bean实例(无参构造)
2 为bean的属性设置值和对其他bean引用(调用set方法)
3 执行初始化的方法
4 获取创建bean实例对象
com.sdl.spring.bean.Orders@4e50c791
5 执行销毁的方法
```





**4、bean的后置处理器: bean生命周期有七步**

​	(1) 通过构造器创建bean实例(无参构造)

​	(2) 为bean的属性设置值和对其他bean引用(调用set方法)

​	**(3) 把bean实例传递bean后置处理器的方法**

​    (4) 调用bean的初始化的方法(需要进行配置)

​	**(5) 把bean实例传递bean后置处理器的方法postProcessAfterInitialization**

​    (6) bean可以使用了(对象获取到了)

​	(7) 当容器关闭时候,调用bean的销毁方法(需要进行配置销毁的方法)

**5、添加后置处理器效果**

​	(1) 创建类, 实现接口BeanProcessor, 创建后置处理器

```java
package com.atguigu.spring5.bean;

public class Orders {

    //无参数构造
    public Orders() {
        System.out.println("第一步 执行无参数构造创建bean实例");
    }

    private String oname;
    public void setOname(String oname) {
        this.oname = oname;
        System.out.println("第二步 调用set方法设置属性值");
    }

    //创建执行的初始化的方法
    public void initMethod() {
        System.out.println("第三步 执行初始化的方法");
    }

    //创建执行的销毁的方法
    public void destroyMethod() {
        System.out.println("第五步 执行销毁的方法");
    }
}

```

MyBeanPost

```java
package com.sdl.spring.bean;

import org.springframework.beans.BeansException;
import org.springframework.beans.factory.config.BeanPostProcessor;

public class MyBeanPost implements BeanPostProcessor{
	
	@Override 
    public Object postProcessBeforeInitialization(Object bean, String beanName) throws BeansException {
        System.out.println("在初始化之前执行的方法");
        return bean;
    }
    @Override
    public Object postProcessAfterInitialization(Object bean, String beanName) throws BeansException {
        System.out.println("在初始化之后执行的方法");
        return bean;
    }

}

```

bean4.xml

```java
<?xml version="1.0" encoding="UTF-8"?>
<beans xmlns="http://www.springframework.org/schema/beans"
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xmlns:p="http://www.springframework.org/schema/p"
       xmlns:util="http://www.springframework.org/schema/util"
       xsi:schemaLocation="http://www.springframework.org/schema/beans http://www.springframework.org/schema/beans/spring-beans.xsd
       http://www.springframework.org/schema/util http://www.springframework.org/schema/util/spring-util.xsd">

   	<bean id = "orders" class = "com.sdl.spring.bean.Orders" init-method = "initMethod" destroy-method = "destroyMethod">
   		<property name="oname" value = "手机"></property>
   	</bean>
   	
   	<!--配置后置处理器-->
    <bean id="myBeanPost" class="com.sdl.spring.bean.MyBeanPost"></bean>
   	
</beans>
```



```java
Output
1 通过构造器创建bean实例(无参构造)
2 为bean的属性设置值和对其他bean引用(调用set方法)
在初始化之前执行的方法
3 执行初始化的方法
在初始化之后执行的方法
4 获取创建bean实例对象
com.sdl.spring.bean.Orders@394df057
5 执行销毁的方法

```

### IOC操作Bean管理(xml自动装配)

****

1、什么是自动装配

​	根据指定装配规则(属性名称或者属性类型), Spring自动将匹配的属性值进行注入

2、演示自动装配过程

​	(1) 根据属性名称自动注入

bean5.xml

```java
<?xml version="1.0" encoding="UTF-8"?>
<beans xmlns="http://www.springframework.org/schema/beans"
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xmlns:p="http://www.springframework.org/schema/p"
       xmlns:util="http://www.springframework.org/schema/util"
       xsi:schemaLocation="http://www.springframework.org/schema/beans http://www.springframework.org/schema/beans/spring-beans.xsd
                           http://www.springframework.org/schema/util http://www.springframework.org/schema/util/spring-util.xsd">

    <!--实现自动装配
        bean标签属性autowire，配置自动装配
        autowire属性常用两个值：
            byName根据属性名称注入 ，注入值bean的id值和类属性名称一样
            byType根据属性类型注入
    -->
    <bean id="emp" class="com.sdl.spring.autowire.Emp" autowire="byName">
        <!-- <property name="dept" ref="dept"></property> -->
    </bean>
        
    <bean id="dept" class="com.sdl.spring.autowire.Dept"></bean>
byType方式:
	当有两种想用类的对象时,根据byType则会报错,因为两个Type都符合,无法装配
    如下:
    <bean id="emp" class="com.sdl.spring.autowire.Emp" autowire="byType">
        <!-- <property name="dept" ref="dept"></property> -->
    </bean>
	<bean id="dept1" class="com.sdl.spring.autowire.Dept"></bean>
    <bean id="dept2" class="com.sdl.spring.autowire.Dept"></bean>
</beans>
```

TestDemo

```java
@Test
    public void test4() {
        ApplicationContext context =
                new ClassPathXmlApplicationContext("bean5.xml");
        Emp emp = context.getBean("emp", Emp.class);
        System.out.println(emp);
    }
```

Emp: Emp中的属性名称和bean中自动加载的名称一样,都是dept,因此能够匹配

```java
package com.sdl.spring.autowire;

public class Emp {
	private Dept dept;
    public void setDept(Dept dept) {
        this.dept = dept;
    }

    @Override
    public String toString() {
        return "Emp{" +
                "dept=" + dept +
                '}';
    }

    public void test() {
        System.out.println(dept);
    }
}

```

Dept

```javapackage com.sdl.spring.autowire;
public class Dept {
	@Override
    public String toString() {
        return "Dept{}";
    }
}
```

```java
output:
Emp{dept=Dept{}}
解释:如果不重写toString直接输出某对象,输出的是该对象的地址
重写toString则会输出重写后的toString结果
```

### IOC操作Bean管理(外部属性文件)

****

**1、直接配置数据库信息**

​	(1)配置德鲁伊连接池

​	(2)引入德鲁伊连接池依赖jar包

```java
<!--直接配置连接池-->
    <bean id="dataSource" class="com.alibaba.druid.pool.DruidDataSource">
        <property name="driverClassName" value="com.mysql.jdbc.Driver"></property>
        <property name="url" value="jdbc:mysql://localhost:3306/userDb"></property>
        <property name="username" value="root"></property>
        <property name="password" value="root"></property>
    </bean>
```

**2、引入外部属性文件配置数据库连接池**

​	(1) 创建外部属性文件, property格式文件, 写数据库 信息（**jdbc.properties**）

```java
jdbc.properties
prop.driverClass=com.mysql.jdbc.Driver
prop.url=jdbc:mysql://localhost:3306/userDb
prop.userName=root
prop.password=root
```

（2）把外部 properties 属性文件引入到 spring 配置文件中 —— 引入 context 名称空间

```java
bean6.xml
<?xml version="1.0" encoding="UTF-8"?>
<beans xmlns="http://www.springframework.org/schema/beans"
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xmlns:p="http://www.springframework.org/schema/p"
       xmlns:util="http://www.springframework.org/schema/util"
       xmlns:context="http://www.springframework.org/schema/context"
       xsi:schemaLocation="http://www.springframework.org/schema/beans http://www.springframework.org/schema/beans/spring-beans.xsd
                           http://www.springframework.org/schema/util http://www.springframework.org/schema/util/spring-util.xsd
                           http://www.springframework.org/schema/context http://www.springframework.org/schema/context/spring-context.xsd">
    <!--引入外部属性文件-->
    <context:property-placeholder location="classpath:jdbc.properties"/>

    <!--配置连接池-->
    <bean id="dataSource" class="com.alibaba.druid.pool.DruidDataSource">
        <property name="driverClassName" value="${prop.driverClass}"></property>
        <property name="url" value="${prop.url}"></property>
        <property name="username" value="${prop.userName}"></property>
        <property name="password" value="${prop.password}"></property>
    </bean>

</beans>
```

**IOC操作Bean管理（基于注解方法）**

1、什么是注解

​	（1）注解是代码的特殊标记，格式：@注解名称（属性名称=属性值，属性名称=属性值...）

​	（2）使用注解，注解作用在类上面，方法上面，属性上面

​	（3）使用注解目的：简化xml配置

2、Spring针对Bean管理中创建对象提供注解

​	（1）@Componet

​	（2）@Service

​	（3）@Controller

​	（4）@Repository

*上面四个注解功能是一样的，都可以用来创建bean实例

3、基于注解方式实现对象创建

第一步：引入依赖

spring-aop-5.2.6

第二步：开启组件扫描

```java
<!--开启组件扫描
        1、如果扫描多个包，多个包使用逗号隔开
        com.sdl.spring.dao, com.sdl.spring.service
        2、扫描包上层目录
        com.sdl
    -->
    <context:component-scan base-package="com.sdl"></context:component-scan>
```

第三步：创建类，在类上面添加创建对象注解

```java
package com.sdl.spring.service;
import com.sdl.spring.dao.UserDao;
//在注解里面value属性值可以省略不写，
//默认值是类名称，首字母小写
//UserService -- userService
@Component(value = "userService")  //<bean id="userService" class=".."/>
public class UserService {
    private UserDao userDao;

    public void add() {
        System.out.println("service add......."+name);
        userDao.add();
    }
}
```

**4、开启组件扫描细节配置**

1、context:include-filter：设置扫描哪些内容

```java
<!--开启组件扫描
    1、如果扫描多个包，多个包使用逗号隔开
    com.sdl.spring.dao, com.sdl.spring.service
    2、扫描包上层目录
    com.sdl
-->
<context:component-scan base-package="com.sdl"></context:component-scan>

<!--示例1
    use-default-filters="false" 表示现在不适用默认filter，自己配置filter
    context:include-filter 设置扫描哪些内容，只会去com.sdl中扫描带有Controller注解的类
-->
<context:component-scan base-package="com.sdl" use-default-filters="false">
    <context:include-filter type="annotation"
                            expression="org.springframework.stereotype.Controller"/>
</context:component-scan>
```



2、context:exclude-filter：设置哪些内容不去扫描

```java
<!--示例2
    下面配置扫描包有哪些内容
    context:exclude-filter:设置哪些内容不去扫描
    即:com.sdl包中除了带有Controller注解的类都进行扫描
-->
<context:component-scan base-package="com.sdl">
    <context:exclude-filter type="annotation"
                            expression="org.springframework.stereotype.Controller"/>
</context:component-scan>
```

**5、基于注解方式实现属性注入**

（1）@AutoWired：根据属性类型进行自动装配

（2）@Qualifier：根据属性名称进行注入

（3）@Resource：可以根据类型注入，也可以根据名称注入

（4）@Value：注入普通类型属性

****

注解的意义：

（1）注解是代码的特殊标记，格式：@注解名称（属性名称=属性值，属性名称=属性值...）

（2）使用注解，注解作用在类上面，方法上面，属性上面

（3）使用注解目的：简化xml配置

****

（1）@AutoWired：根据属性类型进行自动装配

第一步：把service和dao对象创建，在service和dao类添加创建对象注解

```java
package com.sdl.spring.dao;

public interface UserDao {
    public void add();
}
```

```java
package com.sdl.spring.dao;
import org.springframework.stereotype.Repository;
@Repository(value = "userDaoImpl1")
public class UserDaoImpl implements UserDao {
    @Override
    public void add() {
        System.out.println("dao add.....");
    }
}
```



第二步：在service注入dao对象，在service类添加dao类型属性，在属性上面使用注解

```java
@Service
public class UserService {

    @Value(value = "abc")
    private String name;

    //定义dao类型属性
    //不需要添加set方法
    //添加注入属性注解
    @Autowired  //根据类型进行注入
    private UserDao userDao;

    public void add() {
        System.out.println("service add......."+name);
        userDao.add();
    }
```

（2）@Qualifier：根据属性名称进行注入

这个Qualifier注解的使用，**和上面@AutoWired一起使用**

```java
@Service
public class UserService {

    @Value(value = "abc")
    private String name;

    //定义dao类型属性
    //不需要添加set方法
    //添加注入属性注解
    @Autowired  //根据类型进行注入
    @Qualifier(value = "userDaoImpl1") //根据名
    private UserDao userDao;

}
```



（zhong3）@Resource：可以根据类型注入，也可以根据名称注入

```java
@Resource  //根据类型进行注入
@Resource(name = "userDaoImpl1")  //根据名称进行注入
    private UserDao userDao;
```



（4）@Value：注入普通类型属性

```java
@Value(value = "abc")
private UserDao userDao;
```

**6、完全注解开发**

（1）创建配置类，替代xml配置文件

```java
原写法：
<context:component-scan base-package="com.sdl"></context:component-scan>
```

```java
等价于：
@Configuration  //作为配置类，替代xml配置文件
@ComponentScan(basePackages = {"com.sdl"})
public class SpringConfig {

}
```

（2）编写测试类

```java
加载配置类SpringConfig中的内容
ApplicationContext context
        = new AnnotationConfigApplicationContext(SpringConfig.class);
```

```
@Test
public void testService2() {
    //加载配置类
    ApplicationContext context
            = new AnnotationConfigApplicationContext(SpringConfig.class);
    UserService userService = context.getBean("userService", UserService.class);
    System.out.println(userService);
    userService.add();
}
```

