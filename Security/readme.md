# Work on security

## DVWA(Damn Vulnerable Web App)

We first use DVWA(Damn Vulnerable Web App) as a testing platform to show the basics of burp. It is a PHP/MySQL web application which is damn vulnerable. Its main goals are to be an aid for security professionals to test their skills and tools in a legal environment, help web developers better understand the processes of securing web applications and aid teachers/students to teach/learn web application security in a class room environment.

However, when we fly in the face of all the set-up information on the [DVWA site](http://www.dvwa.co.uk/), we decide to go down the docker route and it is quite simple, thanks to the inspiring work by [infoslack](https://github.com/infoslack/docker-dvwa), even more elegant than the recommended way using [XAMPP(Apache + MariaDB + PHP + Perl)](https://www.apachefriends.org/index.html).

So how do we realize it? There are only two steps.

### install Docker

Details can be found from this [tutorial](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-16-04). Briefly speaking, just do the following steps. Our environment is `Ubuntu 18.04.1 LTS`.

- install Docker from the official Docker repository.

```shell
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt-get update
apt-cache policy docker-ce
sudo apt-get install -y docker-ce
```

- valid docker installation

```shell
sudo systemctl status docker
```

### work with DVWA docker image

- pull docker image

```shell
sudo docker pull infoslack/dvwa
sudo docker run -d -p 80:80 infoslack/dvwa
```
You may also change random mysql password to self-defined one, just type:

```shell
sudo docker run -d -p 80:80 -p 3306:3306 -e MYSQL_PASS="mypass" infoslack/dvwa
```

- learn DVWA

Just browse to *localhost(127.0.0.1)*, and hit `Create/Reset Database`. Enjoy it :)

- stop DVWA service

```shell
sudo docker ps
sudo docker stop <container-id>
```

## Introduction to Web app attacks

- [ ] TODO

