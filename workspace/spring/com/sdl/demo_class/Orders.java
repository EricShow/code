package com.sdl.demo_class;

public class Orders {
	private String oname;
	private String address;
	public Orders(String oname, String address) {
		this.oname = oname;
		this.address = address;
	}
	public String getOrderName() {
		return this.oname;
	}
}
