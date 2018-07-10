#!/usr/bin/env bash

sudo groupadd docker
sudo gpasswd -a ${USER} docker
sudo service docker restart
18.04+ with snap:
sudo systemctl restart snap.docker.dockerd
# reboot