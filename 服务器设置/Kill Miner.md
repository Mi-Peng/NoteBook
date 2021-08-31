### Kill Miner

---

> 简介：服务器密码登录，被PhoenixMiner黑入挖矿，且更改了系统文件如find，chattr

##### 背景

`top` 命令：没有cpu占用率高的不明进程

`nvidia-smi` 命令：没有相关程序运行，但每张显卡有4985MB的显存占用，且显卡占用率持续100%，且自身代码运行速度极慢；由于服务器刚装好系统没有配置密钥，初步判断是被挖矿代码黑了。

<img src="./figs/Kill Miner_nvidia-smi.png" width=70%>

##### 诊断过程

首先切换至root用户

```bash
sudo su 
```

> ctrl + d 切回之前账号

首先使用 `netstat -napt` 查看网络端口，通过端口挨个排查发现一个可疑的ssh登录：通过命令 `systemctl status 进程号` 查看该进程历史记录：

<img src="./figs/Kill Miner_systemctl.png">

其中的zhl@59.77.17.59并不是我们实验室熟知的ip，基本确诊服务器被黑。

与此同时在某个账号目录下的anaconda3/miniconda3 执行命令`ll -rt`，发现有root权限创建的sbin文件夹，文件夹下有config.txt python* eptools.txt三个文件夹，vim打开之后赫然写着"PhoenixMiner", 确诊服务器被黑

##### 治疗过程

首先切到root用户：

```bash
sudo su
```

> ctrl + d 切换回之前账号

去到proc目录，查询python进程

```bash
root@ubuntu18: /home/mip$ cd /proc
root@ubuntu18: /proc$ find . -name python
bash:/usr/bin/find: Permission denied.
```

但此时报错：`bash:/usr/bin/find: Permission denied.` 我们使用find命令实际上调用/usr/bin下的find文件，但find文件的权限被挖矿脚本修改了，我们首先尝试 `chmod` 修改find文件

```bash
root@ubuntu18: /usr/bin$ chmod 755 find
chmod: changing permissions of '/usr/bin/find': Operation not permitted
```

但报错： `chmod: changing permissions of '/usr/bin/find': Operation not permitted `，简而言之，修改find文件权限的权限被取消。

使用lsattr命令查看文件属性

```bash
root@ubuntu18: /usr/bin$ lsattr find
-u-ai---------e--- find
```

可以看到find文件的属性中有i与a，其中i参数表示，防止系统中该文件被修改，也就是我们之前`chmod` 命令不能工作的原因。

我们接下来使用 chattr命令修改文件权限（去掉i就好了嘛）

```bash
root@ubuntu18: /usr/bin$ chattr -i find
Command chattr not found
```

显示我们没有chattr命令，与另一台没有被黑，正常工作的服务器做对比，正常服务器的/usr/bin下有 chattr文件，我们被黑掉的服务器的/usr/bin 下没有chattr文件，自然没有chattr指令，我们把chattr文件从另一台服务器上复制过来，到该服务器下的/usr/bin 目录下

```bash 
root@ubuntu18: /usr/bin$ chmod 755 chattr
```

此时我们就把被黑客删掉的chattr命令找了回来，使用chattr命令修改find文件属性

```bash
root@ubuntu18: /usr/bin$ chattr -i -a find
```

我们就拥有了修改find文件的权限，修改find文件：

```bash
root@ubuntu18: /usr/bin$ chmod 755 find
```

然后回到proc目录下查找所有python脚本：

```bash
root@ubuntu18: /usr/bin$ cd /proc
root@ubuntu18: /proc$ find . -name python
```

返回若干进程，kill掉相应挖矿进程

```bash 
root@ubuntu18: /proc$ kill -9 xxxxx
```

接下来删除账号目录下的anaconda3/miniconda3下的sbin目录中的三个文件，此时rm命令遇到同样问题，再次使用lsattr命令查看文件发现 sbin文件夹 +i 上锁，同样使用chattr命令解锁，rm相应文件即可。

##### 附加：

查看定时任务：

```bash 
crontab -l
```

删除定时任务：

```bash
crontab -r
```

记得关闭密码登录开启密钥登录，具体详见另一篇教程。

