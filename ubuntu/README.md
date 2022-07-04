# Ubuntu环境搭建

## 安装Ubuntu20.04LTS WIFI6网卡

现在新的主板往往会搭载无限网卡，但是Ubuntu20.04LTS无法正确安装这些网卡驱动，所以会导致无线网卡无法使用，那么如何处理呢？

### Wifi6 AX211无线网卡驱动安装

```
$ wget https://launchpad.net/ubuntu/+archive/primary/+files/backport-iwlwifi-dkms_9858-0ubuntu3_all.deb
$ sudo apt install ./backport-iwlwifi-dkms_9858-0ubuntu3_all.deb
$ sudo reboot
```

如果想安装最先的Wifi6驱动，那么也时可以考虑下载无线网卡的最新版本

```
$ wget https://launchpad.net/ubuntu/+archive/primary/+files/backport-iwlwifi-dkms_9858-0ubuntu5_all.deb
```

### Wifi6 AX210无线网卡驱动安装

```
$ wget https://launchpad.net/ubuntu/+archive/primary/+files/backport-iwlwifi-dkms_8324-0ubuntu3~20.04.4_all.deb
$ sudo apt install ./backport-iwlwifi-dkms_8324-0ubuntu3~20.04.4_all.deb
$ sudo reboot
```

## Ubuntu Desktop Images

- [ubuntu-22.04-desktop-amd64.iso](https://releases.ubuntu.com/22.04/ubuntu-22.04-desktop-amd64.iso)

    ```
    $ wget (https://releases.ubuntu.com/22.04/ubuntu-22.04-desktop-amd64.iso
    ```

- [ubuntu-20.04.4-desktop-amd64.iso](https://releases.ubuntu.com/20.04/ubuntu-20.04.4-desktop-amd64.iso)

    ```
    $ wget https://releases.ubuntu.com/20.04/ubuntu-20.04.4-desktop-amd64.iso
    ```

- [ubuntu-18.04.6-desktop-amd64.iso](https://releases.ubuntu.com/18.04/ubuntu-18.04.6-desktop-amd64.iso)

    ```
    $ wget https://releases.ubuntu.com/18.04/ubuntu-18.04.6-desktop-amd64.iso
    ```

- [ubuntu-16.04.7-desktop-amd64.iso](https://releases.ubuntu.com/16.04/ubuntu-16.04.7-desktop-amd64.iso)

    ```
    $ wget https://releases.ubuntu.com/16.04/ubuntu-16.04.7-desktop-amd64.iso
    ```

- [ubuntu-14.04.6-desktop-amd64.iso](https://releases.ubuntu.com/14.04/ubuntu-14.04.6-desktop-amd64.iso)

    ```
    $ wget https://releases.ubuntu.com/14.04/ubuntu-14.04.6-desktop-amd64.iso
    ```



## 参考

* [ubuntu 20.04 LTS driver intel WI-FI 6E AX211 160MHZ](https://askubuntu.com/questions/1398392/ubuntu-20-04-lts-driver-intel-wi-fi-6e-ax211-160mhz)
* [backport-iwlwifi-dkms package in Ubuntu](https://launchpad.net/ubuntu/+source/backport-iwlwifi-dkms)
* [Ubuntu 20.04 LTS Intel Wi-Fi 6E AX210 Driver doesn't work after update](https://askubuntu.com/questions/1400376/ubuntu-20-04-lts-intel-wi-fi-6e-ax210-driver-doesnt-work-after-update)
* [These releases of Ubuntu are available](https://releases.ubuntu.com)