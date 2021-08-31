Ubuntu apt更新源

Ubuntu apt vim

Ubuntu apt net-tools

### [Ubuntu 挂载硬盘](https://blog.csdn.net/zhengchaooo/article/details/79500075)

查看当前所有磁盘信息 `sudo fdisk -l`

![image-20210825112017192](image-20210825112017192.png)

使用parted命令挂载大于2TB的硬盘 `parted /dev/sdb`

`mklabel gpt`

`print`

`mkpart primary 0KB 32TB`

`y`

`i`

`quit`退出parted

`sudo mkfx.ext4 /dev/sdb` 格式化为ext4文件系统

`sudo mount /dev/sdb /home/yourdir` 挂载到dir路径

在/etc/fstab文件追加内容：/dev/sdb /yourdirpath ext4 defaults 0 0

`df -h` 查看

### Ubuntu 安装NVIDIA驱动

`ubuntu-drivers devices`查看推荐驱动，一般返回：

```bash
== /sys/devices/pci0000:00/0000:00:01.0/0000:01:00.0 ==
modalias : pci:v000010DEd00001180sv00001458sd0000353Cbc03sc00i00
vendor : NVIDIA Corporation
model : GP106 [GeForce GTX 1060 6GB]
driver : nvidia-304 - distro non-free
driver : nvidia-340 - distro non-free
driver : nvidia-390 - distro non-free recommended
driver : xserver-xorg-video-nouveau - distro free builtin
```

`sudo ubuntu-drivers autoinstall`会安装上述中 recommended的驱动

安装之后`sudo reboot`

### Ubuntu 安装配置CUDA

卸载CUDA:

卸载程序在`/usr/local/cuda-xx.x/bin`下，需要注意的是cuda10.0及之前的版本卸载程序名为`uninstall_cuda_xx.x.pl`，而cuda10.1及之后的版本卸载程序名为`cuda-uninstaller`。

**CUDA10.0及以下**

```bash
cd /usr/local/cuda-xx.x/bin/
sudo ./uninstall_cuda_xx.x.pl
sudo rm -rf /usr/local/cuda-xx.x
```

**CUDA10.0及以上**

```bash
cd /usr/local/cuda-xx.x/bin/
sudo ./cuda-uninstaller
sudo rm -rf /usr/local/cuda-xx.x
```

安装：

https://developer.nvidia.com/cuda-toolkit-archive

### Ubuntu新建用户并根据用户创建密钥

存在一台服务器假设ip为**xx.xx.xx.15**，OS为Ubuntu16.04.服务器上存在用户A，服务器只允许密钥登录。通过用户A的用户密码与密钥登录进入**xx.xx.xx.15**，首先新建用户B：

```bash
A@ubuntu:~$ sudo adduser B sudo
```

> Ubuntu18.04: 
>
> 新建用户 -r : 建立系统账号 -m: 自动建立用户登入目录 -s: 指定用户登入后所使用shell
>
> ```bash
> A@ubuntu:~$ sudo useradd -r -m -s /bin/bash yourusername
> ```
>
> 为新建的用户设置密码
>
> ```bash
> A@ubuntu:~$ sudo passwd yourusername
> ```
>
> 为新建的用添加sudo权限
>
> ```bash
> A@ubuntu:~$ sudo vim /etc/sudoers
> ```
>
> 添加  `yourusername ALL=(ALL:ALL) ALL`  wq！保存退出

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
A@ubuntu:~$ chown A id_rsa
```

然后通过Xshell下载私钥

重启ssh服务

```bash
A@ubuntu:~$ sudo service sshd restart
```

寄！

### 安装Anaconda3

清华镜像源 https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/ 下载安装包 Anaconda3-5.3.1-Linux-x86_64.sh 通过winscp上传到服务器

```bash
B@ubuntu:~$bash Anaconda3-5.3.1-Linux-x86_64.sh
```

关于配置环境变量：

```bash
B@ubuntu:~$vim ~/.bashrc
```

在文件最后写入

```bash
export PATH=~/anaconda3/bin:$PATH
```

重启环境变量

```bash
source ~/.bashrc
```

寄！

### Tmux

下载 tmux：

```bash
B@ubuntu:~$ sudo apt install tmux
```

新建会话：

```bash
B@ubuntu:~$ tmux new -s crl
```

或者重新链接进已经建立好的会话

```bash
B@ubuntu:~$ tmux a -t crl
```

设置鼠标操作：

```bash
set-option -g mouse on
```

### 下载更新配置GIT

`sudo apt-get install git`

`git config --list` 查看用户 （新服务器新用户为空）

```bash
git config --global user.name "your name here"
 
git config --global user.email "your email@example.com"
```

设置用户如上

`git config --list` 再次查看：

```bash
user.name=XXX
user.email=XXXX@XXX
```

`ssh-keygen -C XXXX@XXXX  `   生成github邮箱密钥

`cat ~/.ssh/id_rsa.pub`  将生成的公钥复制到github网站，setting，SSH and GPG keys即可

### Anaconda切换清华源

```bash
B@ubuntu:~$ conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
B@ubuntu:~$ conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
B@ubuntu:~$ conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
B@ubuntu:~$ conda config --set show_channel_urls yes
```

### Anaconda新建虚拟环境

```bash
B@ubuntu:~$ conda create -n crl python=3.8
```

激活虚拟环境：

```bash
B@ubuntu:~$ source activate
(base) B@ubuntu:~$ conda activate crl
```

下载相应第三方库

### 常用第三方库

* **pytorch**: `conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 pytorch` 需要`cat /usr/local/cuda/version.txt `找到对应CUDA版本到官网https://pytorch.org/找到相应命令
* **dali**：(for CUDA10.0)`pip install--extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda100`，详见官网https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html
* **hydra**: `pip install hydra-core --upgrade`
* **numpy**: pytorch 安装中包含numpy
* **matplotlib**: `conda install matplotlib`
* **scikit-learn**:`pip3 install -U scikit-learn`
* **scipy**: sklearn 安装中包含scipy
* **tqdm**:`pip install tqdm`
* **pyyaml**:（目前不再使用）
* **einops**: `pip install einops`
* **lars**: `pip install torchlars`该第三方库教程见https://github.com/kakaobrain/torchlars （该包可能由于一台服务器上有多个CUDA版本而报错）

### 离线安装pip包

通过`pip download --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda110 `安装CUDA10版本对应dali；

```bash
Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple, https://developer.download.nvidia.com/compute/redist
Collecting nvidia-dali-cuda110
  Downloading https://developer.download.nvidia.cn/compute/redist/nvidia-dali-cuda110/nvidia_dali_cuda110-1.2.0-2356513-py3-none-manylinux2014_x86_64.whl (613.6 MB)
     |████████████████████████████████| 613.6 MB 26 kB/s 
Saved ./nvidia_dali_cuda110-1.2.0-2356513-py3-none-manylinux2014_x86_64.whl
Successfully downloaded nvidia-dali-cuda110
```

通过`pip install ./nvidia_dali_cuda110-1.2.0-2356513-py3-none-manylinux2014_x86_64.whl`安装。

寄！

