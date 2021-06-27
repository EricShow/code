Java  jdk重新安装踩坑集锦

1、找到环境变量

​	此电脑-属性-高级系统设置（右侧）-高级-环境变量

​			1、找到Javahome，修改新安装的jdk8

​			2、找到系统变量（下面）中的Path选项调整

​	%JAVA_HOME%\bin到第一位，同时删除Java  oracle环境变量（更新得到的，没用）以及对应目录下的java文件

https://blog.csdn.net/qq_34950682/article/details/94339981?utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7EBlogCommendFromMachineLearnPai2%7Edefault-1.control&dist_request_id=1332031.194.16190084094546241&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7EBlogCommendFromMachineLearnPai2%7Edefault-1.control

这个博客写的很详细