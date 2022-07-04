# VMWare Player环境搭建

## Ubuntu22.04LTS

安装好VMWare Player后需要安装它的模块配置软件，默认的方法会失败，看上去是因为Ubuntu22.04LTS太新，相关的配置文件比较就的原因，所以需要手工的安装，如下所示：

```
$ sudo apt update
$ sudo apt install build-essential
$ sudo sh VMware-Player-Full-16.2.3-19376536.x86_64.bundle
$ wget https://github.com/mkubecek/vmware-host-modules/archive/refs/tags/w16.2.3-k5.18.tar.gz
$ tar xvzf w16.2.3-k5.18.tar.gz
$ cd vmware-host-modules-w16.2.3-k5.18
$ tar -cf vmmon.tar vmmon-only
$ tar -cf vmnet.tar vmnet-only
$ sudo cp -v vmmon.tar vmnet.tar /usr/lib/vmware/modules/source/
$ sudo vmware-modconfig --console --install-all
```

在成功完成上面的命令后，既可以运行VMWare Player。

## 如何卸载VMWare Player？

```
$ sudo vmware-installer -u vmware-player
```

## 在远程桌面中运行VMWare Player

在远程桌面中运行VMWare会失败，大致的失败信息：`ISBRendererComm: Lost connection to mksSandbox (2878)`,从https://communities.vmware.com/的讨论中，有两种方法：

- 降级VMWare Player，比如说从16.x -> 15.5.6
- 在xxx.vmx文件中增加`mks.sandbox.socketTimeoutMS = "200000"`

对于第一种方法，笔者没有验证，因为总有一个版本可以运行，并且不断尝试的时间代价很大。对于第二种方法，笔者尝试了一下，没有成功。从错误信息的，笔者猜想可能是3D Graphic的原因，所以将它disable后即可运行，如下所示：

```
$ nano xxx.vmx
mks.enable3d = "FALSE"
```



## 参考

- [VMware Workstation 16 Player](https://www.vmware.com/products/workstation-player/workstation-player-evaluation.html)
- [VMware 16.2.3 not working on Ubuntu 22.04 LTS](https://communities.vmware.com/t5/VMware-Workstation-Pro/VMware-16-2-3-not-working-on-Ubuntu-22-04-LTS/td-p/2905535)
- [Install VMware Workstation Player on Ubuntu 22.04 LTS](https://www.how2shout.com/linux/install-vmware-workstation-player-on-ubuntu-22-04-lts/)
- [mkubecek/vmware-host-modules](https://github.com/mkubecek/vmware-host-modules)
- [VM Crash 16.2.1 build-18811642](https://communities.vmware.com/t5/VMware-Workstation-Pro/VM-Crash-16-2-1-build-18811642/td-p/2877469)
- [cant boot the vm anymore](https://communities.vmware.com/t5/VMware-Workstation-Pro/cant-boot-the-vm-anymore/td-p/2879014)
- [VMware Workstation unrecoverable error: (mks) ISBRendererComm: Lost connection to mksSandbox (2878)" at end of disk shrink on Windows 10 host Fedora guest #561](https://github.com/vmware/open-vm-tools/issues/561)