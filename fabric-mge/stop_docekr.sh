sudo docker stop $(sudo docker ps -a| grep 'hyperledger' | awk '{print $1}')