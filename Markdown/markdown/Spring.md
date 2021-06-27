### Spring

* 案例一：使用Spring的IOC技术完成客户保存的功能

* 需求分析：

  * 使用Spring的IOC技术完成客户的保存功能

* 技术分析之Spring框架的概述和入门

  * 开源
  * 为了解决企业应用开发的复杂性而创建
  * 优势：分层架构，分层架构允许使用者选择使用哪一个组件，同时为 J2EE 应用程序开发提供继承的框架
  * 核心：控制反转（IoC）和面向切面（AOP)，轻量级开发框架
  * EE开发分三层结构：
    * web层（Struts2）            --Spring MVC
    * 业务层（Spring）             --Bean管理：(IOC)
    * 持久层（Hibernate）       --Spring的JDBC模板，ORM模板用于整合其他的持久层框架
  * 一站式开发：Spring全能，整合其他框架

* 技术分析之Spring框架的特点

  * 方便解耦，简化开发
  * AOP编程的支持：* Spring提供面向切面编程，可以方便的实现对程序进行权限拦截、运行监控等功能
  * 声明式事务的支持：* 只需要通过配置就可以完成对事务的管理，而无需手动编程
  * 方便程序的测试：* Spring对Junit4支持，可以通过注解方便的测试Spring程序
  * 方便集成各种优秀框架：* Spring不排斥各种优秀的开源框架，其内部提供了对各种优秀框架（如：Struts2、Hibernate、MyBatis、Quartz等）的直接支持
  * 降低JavaEE API的使用难度：* Spring 对JavaEE开发中非常难用的一些API（JDBC、JavaMail、远程调用等），都提供了封装，使这些API应用难度大大降低

* 技术分析之Spring框架的IOC核心功能快速入门

  *  * docs		  -- API和开发规范
    * libs		    -- jar包和源码
    * schema	-- 约束

  * 引入Spring框架IOC核心功能需要的具体的jar包
  		* Spring框架的IOC的功能，那么根据Spring框架的体系结构图能看到，只需要引入如下的jar包
  			* Beans
  		* Core
  		* Context
  		* Expression Language

* 组成 

![img](https://img.jbzj.com/file_images/article/201711/20171127110222693.jpg?2017102711233)

* 拓展
  * Spring Boot
    * 一个快速开发的脚手架
    * 基于SpringBoot可以快速地开发单个微服务
    * 约定大于配置
  * Spring Cloud
    * 它是基于SpringBoot实现的
  * 大部分公司都是SpringBoot进行快速开发，学习SpringBoot的前提，需要完全掌握Spring及SpringMVC

### 2、IOC理论推导

​	1、UserDao接口

​	2、UserDaoImpl实现类

​	3、UserService 业务接口

​	4、UserService 业务实现类

