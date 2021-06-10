package com.sdl.demo_test.service;

import com.sdl.demo_test.dao.UserDao;
import com.sdl.demo_test.dao.UserDaoImpl;

public class UserService {
	
	//创建UserDao类型属性，生成set方法
	private UserDao userDao;
	
	public void setUserDao(UserDao userDao) {
		this.userDao = userDao;
	}
	
	public void add() {
		System.out.println("Service add .............");
		userDao.update();
		/* 原始方式
		 * //创建UserDao对象 
		 * UserDao userDao = new UserDaoImpl(); 
		 * userDao.update();
		 */
	}


	
}
