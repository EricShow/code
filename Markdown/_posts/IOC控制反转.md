## IOC控制反转

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

## 二、IOC容器-Bean管理

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

