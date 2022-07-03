# Docker环境搭建

## 如何改变docker images存放路径？

```
$ sudo service docker stop
$ sudo nano /etc/docke/daemon.json
{
  "data-root": "/path/to/your/docker",
  ...
}
$ sudo rsync -aP /var/lib/docker/ /path/to/your/docker
$ sudo mv /var/lib/docker /var/lib/docker.old
$ sudo service docker start
$ sudo rm -rf /var/lib/docker.old
```



## 参考

- [How to change docker root data directory](https://tienbm90.medium.com/how-to-change-docker-root-data-directory-89a39be1a70b)
- [HOW TO MOVE DOCKER DATA DIRECTORY TO ANOTHER LOCATION ON UBUNTU](https://www.guguweb.com/2019/02/07/how-to-move-docker-data-directory-to-another-location-on-ubuntu/)