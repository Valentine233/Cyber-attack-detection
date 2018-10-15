## Reference
Ubuntu上安装Docker: https://docs.docker.com/install/linux/docker-ce/ubuntu/

ELK官方文档：https://elk-docker.readthedocs.io/

Docker部署ELK的博文：https://juejin.im/post/5b80b37f6fb9a019d53e8b34 (本文建立在此基础上)

ELK介绍：http://docs.flycloud.me/docs/ELKStack/logstash/index.html

原文：https://blog.csdn.net/zmx1996/article/details/83039762?utm_source=copy 

## 快速搭建步骤

### 拉取ELK镜像：
```
docker pull sebp/elk
```

### 使用docker命令验证是否拉取镜像成功：
```
docker images
```

### 运行容器：
```
docker run -p 5601:5601 -p 9200:9200 -p 5044:5044 -v /Users/song/dockerFile:/data -it -d --name elk sebp/elk
```
-p 参数为端口映射，格式为 主机端口：容器端口

上述命令表示，ElasticSearch端口为9200， kibana端口为5601， logstash端口为5044；

-v 参数表明，将主机目录/Users/song/dockerFile映射到容器的/data目录下

### 查看Kibana:
在浏览器中访问Kibana页面, URL为：

http://localhost:5601/

### 查看ElasticSearch服务：
在浏览器中方巍峨ES页面，URL为：

http://localhost:9200/_search?pretty

### Logstash的配置：
```
vim /Users/song/dockerFile/config/logstash.conf
```

由于Nginx的log数据格式为：
> 183.202.215.225 - - [20/Jul/2017:14:39:34 +0800] "GET /stat.html?site=shuidigzh&action=http://d.ifeng.com/webppa/315/060109/index.shtml&uid=150
0530496249_uwejx13000&url=http://d.ifeng.com/webppa/hd/gdian/061516/index.shtml&referrer=http://d.ifeng.com/webppa/315/060109/index.shtml&rehos
t=d.ifeng.com&dateline=1500532799523 HTTP/1.1" 200 0 "http://d.ifeng.com/webppa/hd/gdian/061516/index.shtml" "183.202.215.225" "Mozilla/5.0 (Li
nux; Android 4.3; S8-701w Build/HuaweiMediaPad; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/53.0.2785.49 MQQBrowser/6.2 TBS/0
43305 Safari/537.36 MicroMessenger/6.5.8.1060 NetType/WIFI Language/zh_CN"

我们事先将收集到的日志文件放入/Users/song/dockerFile/data/logs/access.log

于是配置文件可以写为

```bash
input {
    file {
        path => "/data/logs/access.log"
        start_position => "beginning"
    }
}
}
filter {
    grok {
        match => { "message" => "%{IPORHOST:clientip} - - \[%{HTTPDATE:timestamp}\] \"(?:%{WORD:verb} %{NOTSPACE:request}(?: HTTP/%{NUMBER:httpversion})?|%{DATA:rawrequest})\" %{NUMBER:response} (?:%{NUMBER:bytes}) %{QS:referrer} %{QS:agent}"   }
 
    }
}
output {
    elasticsearch {
        hosts => ["10.90.32.48:9200"]
    }
    stdout {code => rubydebug}
}
```

### 启动logstash:
```
docker exec -it elk bash
```

进入容器之后，启动logstash：
```
/opt/logstash/bin/logstash --path.data /root/ -f /data/config/logstash.conf
```

### Kibana的可视化：
在kibana的界面上

http://127.0.0.1:5601/app/kibana#/management/kibana/index?_g=()

可以在discover界面创建索引

在visualize界面进行可视化（如图标等的绘制）

## 注意！
页面右上角的时间设置，默认似乎是15min之内处理得到的数据，根据需求更改吧……
