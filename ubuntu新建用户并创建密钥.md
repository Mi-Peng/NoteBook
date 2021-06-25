### Background

存在一台服务器假设ip为**xx.xx.xx.15**，OS为Ubuntu16.04.服务器上存在用户A，服务器只允许密钥登录。通过用户A的用户密码与密钥登录进入**xx.xx.xx.15**，首先新建用户B：

```bash
A@ubuntu:~$ sudo adduser B sudo
```

新建用户B之后切换到用户B

```bash
A@ubuntu:~$ su B
```

首先在服务器上为**当前用户**建立密钥对，所以一定要切换到用户B

```bash
B@ubuntu:~$ ssh-keygen -t rsa
Generating public/private rsa key pair.
Enter file in which to save the key (/home/B/.ssh/id_rsa): <== 按 Enter
Created directory '/home/B/.ssh'.
Enter passphrase (empty for no passphrase): <== 输入密钥锁码，或直接按 Enter 留空
Enter same passphrase again: <== 再输入一遍密钥锁码
Your identification has been saved in /root/.ssh/id_rsa. <== 私钥
Your public key has been saved in /root/.ssh/id_rsa.pub. <== 公钥
The key fingerprint is:
0f:d3:e7:1a:1c:bd:5c:03:f1:19:f1:22:df:9b:cc:08 root@host
```

这会在路径下生成一个.ssh文件

```bash
B@ubuntu:~$ cd .ssh
B@ubuntu:.ssh$ cat id_rsa.pub >> authorized_keys
```

以上完成了公钥的安装，为保证链接成功，保证一下文件权限

```bash
B@ubuntu:.ssh$ chmod 600 authorized_keys
B@ubuntu:.ssh$ chmod 700 .
```

将私钥复制回用户A，因为当前无法登录进用户B。

```bash
B@ubuntu:.ssh$ sudo cp id_rsa /home/A 
B@ubuntu:.ssh$ su A
```

返回用户A之后，由于之前复制回来的id_rsa权限属于root用户，所以要对其做权限修改

```bash
A@ubuntu:~$ chown luogen id_rsa
```

然后通过Xshell下载私钥

重启ssh服务

```bash
A@ubuntu:~$ sudo service sshd restart
```

寄！