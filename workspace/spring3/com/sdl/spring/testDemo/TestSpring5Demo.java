package com.sdl.spring.testDemo;
import org.junit.*;
import org.springframework.context.ApplicationContext;
import org.springframework.context.support.ClassPathXmlApplicationContext;

import com.sdl.spring.autowire.*;
import com.sdl.spring.bean.Orders;
import com.sdl.spring.collectiontype.Book;
import com.sdl.spring.collectiontype.Course;
import com.sdl.spring.collectiontype.Stu;
import com.sdl.spring.factorybean.MyBean;
public class TestSpring5Demo {
	@Test
	public void testCollection1() {
		ApplicationContext context = 
				new ClassPathXmlApplicationContext("bean1.xml");
		Stu stu = context.getBean("stu",Stu.class);
		stu.test();
		
	}
}
