---
layout: post
title:  "如何用Docker快速部署Realm Object Server?"
date:   2017-07-25 22:51:14 +0800
categories: 
---

Realm是一款非常优秀的移动端数据库软件, 同时支持iOS和Android平台. 它的后端版本就是Realm Object Server, 是Realm Mobile Platform的组成部分之一. 本文主要介绍如何在Ubuntu平台, 利用Docker来方便快捷的部署Realm Object Server.

#### 后端环境要求
* 阿里云ECS(预装Ubuntu 16.04, 这里Ubuntu的版本只是个巧合, 数据库实际是部署在容器内的);
* Docker;

#### 创建数据库文件夹
通过ssh登录ECS后, 我们需要先创建一个文件夹用来保存数据库的配置/数据文件, 它将扮演共享文件夹的角色, 从容器内外均可访问, 即便容器被删, 宝贵的数据文件并不会丢失.

```Bash
$ cd ~
$ mkdir realmObjectServer
```

#### 安装Docker
Docker社区版(Docker CE)安装指南请前往[https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/#install-docker-ce](https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/#install-docker-ce)

#### 创建Realm Object Server容器
Realm Object Server容器的创建指南请前往[https://github.com/robertwtucker/docker-realm-object-server](https://github.com/robertwtucker/docker-realm-object-server), 该容器最新的镜像(image)配置文件(Dockerfile)在[这里](https://github.com/robertwtucker/docker-realm-object-server/blob/master/Dockerfile).
执行以下命令:

```Bash
$ cd realmObjectServer
$ docker run --name realmObjectServer -d -v $PWD:/var/lib/realm/object-server -p 9080:9080 robertwtucker/realm-object-server
```

备注:

* `dock run`命令用于创建容器;
* `--name`选项用于为容器命名, 该选项如未指定会对容器进行随机命名;
* `-d`选项表示容器启动后进入后台模式, 不会启动交互, detach的缩写;
* `-v`选项用于指定映射文件夹, 即容器内外的共享文件夹, volume的缩写, 冒号前为容器外路径, 冒号后位容器内路径;
* `-p`选项用于指定端口映射, 容器内的端口默认是不公开的, port的缩写, 冒号前为ecs实际端口, 冒号后位容器端口;
* `robertwtucker/realm-object-server`为镜像路径, 这里是一个远程镜像(Docker Hub);

#### 访问Realm Object Server的控制台
容器创建成功后, 可以通过`http://your-domain.com:9080`来访问数据库的Dashboard, 首次访问需要注册admin账号.

#### 下一步干啥?
Realm Object Server部署完毕后, 下一步就可以与移动端进行数据同步了.

