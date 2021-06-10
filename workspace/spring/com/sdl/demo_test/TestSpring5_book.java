package com.sdl.demo_test;
import org.junit.Test;

import org.springframework.context.ApplicationContext;
import org.springframework.context.support.ClassPathXmlApplicationContext;

import com.sdl.demo_class.Book;
public class TestSpring5_book {
	
	@Test
	public void testAdd() {
		//与下面等价BeanFactory context = new ClassPathXmlApplicationContext("bean1.xml");
		ApplicationContext context = 
				new ClassPathXmlApplicationContext("bean1.xml");
	    //User user = context.getBean("user",User.class);
	    Book book = context.getBean("book", Book.class);
		System.out.println("aaa");
		System.out.println(book);
		System.out.println(book.getBname());
		//
	}
}
//输出：
//			aaa
//			com.sdl.spring5.Book@38afe297
//			易筋经