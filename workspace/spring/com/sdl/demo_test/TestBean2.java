package com.sdl.demo_test;

import org.springframework.context.ApplicationContext;
import org.springframework.context.support.ClassPathXmlApplicationContext;

import com.sdl.demo_test.service.UserService;
import org.junit.Test;
public class TestBean2 {
	
	@Test
	public void testAdd() {
		ApplicationContext context = 
				new ClassPathXmlApplicationContext("bean2.xml");
		UserService userService = context.getBean("userService", UserService.class);
		userService.add();
	}
	
}
