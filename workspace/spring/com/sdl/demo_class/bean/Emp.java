package com.sdl.demo_class.bean;

public class Emp {
	private String ename;
	private String gender;
	//员工属于某一个部门，使用对象形式表示
	private Dept dept;
	
	public Dept getDept() {
        return dept;
    }
    public void setDept(Dept dept) {
        this.dept = dept;
    }
    public void setEname(String ename) {
        this.ename = ename;
    }
    public void setGender(String gender) {
        this.gender = gender;
    }
	public void testDemo() {
		System.out.println("name: "+this.ename+"gender: "+this.gender+"dept: "+this.dept);
	}
}
