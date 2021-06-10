package com.sdl.demo_test;
import org.junit.*;
import org.springframework.context.ApplicationContext;
import org.springframework.context.support.ClassPathXmlApplicationContext;

import com.sdl.demo_class.bean.Emp;
public class TestBean3 {
	@Test
	//Bean3对应内部bean
	//Bean4对应级联赋值
	public void testAdd() {
		ApplicationContext context = 
				new ClassPathXmlApplicationContext("bean4.xml");
		Emp emp = context.getBean("emp",Emp.class);
		emp.testDemo();
	}
}
